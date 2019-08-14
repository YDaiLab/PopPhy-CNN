import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score
from utils.popphy_io import get_stat, get_stat_dict
from utils.feature_map_analysis import get_feature_map_rankings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CNN1D():

	def __init__(self, input_len, num_class, config):
		num_kernel = int(config.get('Benchmark', 'NumberKernel_1DCNN'))
		kernel_height = 1
		kernel_width = int(config.get('Benchmark', 'KernelWidth_1DCNN'))
		num_fc_nodes = int(config.get('Benchmark', 'NumFCNodes_1DCNN'))
		num_cnn_layers = int(config.get('Benchmark', 'NumConvLayers_1DCNN'))
		num_fc_layers = int(config.get('Benchmark', 'NumFCLayers_1DCNN'))
		lamb = float(config.get('Benchmark', 'L2_Lambda_1DCNN'))
		drop = float(config.get('Benchmark', 'Dropout_1DCNN'))

		reg = tf.keras.regularizers.l2(lamb)
		self.model = tf.keras.Sequential()

		self.model.add(tf.keras.layers.GaussianNoise(0.01, input_shape=(1,input_len,1)))

		for i in range(0, num_cnn_layers):
			self.model.add(tf.keras.layers.Conv2D(filters=num_kernel, kernel_size=(kernel_height, kernel_width), 
				activation='relu', bias_regularizer=reg, kernel_regularizer=reg, name="conv_"+str(i)))
			self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(1,2)))


		self.model.add(tf.keras.layers.Flatten())
		self.model.add(tf.keras.layers.Dropout(drop))		

		for i in range(0, num_fc_layers):
			self.model.add(tf.keras.layers.Dense(num_fc_nodes, activation='relu', kernel_regularizer=reg, bias_regularizer=reg, name="fc_"+str(i)))
			self.model.add(tf.keras.layers.Dropout(drop))

		self.model.add(tf.keras.layers.Dense(num_class, activation='softmax', kernel_regularizer=reg, bias_regularizer=reg, name="output"))

		self.patience = int(config.get('Benchmark', 'Patience_1DCNN'))
		self.learning_rate = float(config.get('Benchmark', 'LearningRate_1DCNN'))
		self.batch_size = int(config.get('Benchmark', 'BatchSize_1DCNN'))


	def train(self, train, train_weights=[]):
		train_x, train_y = train
		train_x = np.expand_dims(train_x, -1)
		train_x = np.expand_dims(train_x, 1)
		def mcc_metric(y_true, y_pred):
			predicted = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
			true_pos = tf.math.count_nonzero(predicted * y_true)
			true_neg = tf.math.count_nonzero((predicted - 1) * (y_true - 1))
			false_pos = tf.math.count_nonzero(predicted * (y_true - 1))
			false_neg = tf.math.count_nonzero((predicted - 1) * y_true)
			x = tf.cast((true_pos + false_pos) * (true_pos + false_neg) * (true_neg + false_pos) * (true_neg + false_neg), tf.float32)
			return tf.cast((true_pos * true_neg) - (false_pos * false_neg), tf.float32) / tf.sqrt(x)
		
		def auc_metric(y_true, y_pred):
			return tf.numpy_function(roc_auc_score, (y_true, y_pred), tf.double)

		self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss='categorical_crossentropy', 
			metrics=[auc_metric, mcc_metric])

		if len(np.unique(train_y) == 2):
			es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=self.patience, restore_best_weights=True)
		else:
			es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=self.patience, restore_best_weights=True)

		if len(train_weights) == 0:
			self.model.fit(train_x, train_y, batch_size=self.batch_size, verbose=0, epochs=1000, callbacks=[es_cb], validation_split=0.1)
			self.model.fit(train_x, train_y, batch_size=self.batch_size, verbose=0, epochs=10)
		else:
			self.model.fit(train_x, train_y, batch_size=self.batch_size, verbose=0, epochs=1000, callbacks=[es_cb], validation_split=0.1, sample_weight = train_weights)
			self.model.fit(train_x, train_y, batch_size=self.batch_size, verbose=0, epochs=10, sample_weight = train_weights)
		
		return

	def test(self, test):
		test_x, test_y = test
		test_x = np.expand_dims(test_x, -1)
		test_x = np.expand_dims(test_x, 1)

		preds = self.model.predict(test_x)
		stats = get_stat_dict(test_y, preds)
		return preds, stats	


	def destroy(self):
		tf.keras.backend.clear_session()
		return