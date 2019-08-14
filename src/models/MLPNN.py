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

class MLPNN():

	def __init__(self, input_len, num_class, config):
		num_fc_nodes = int(config.get('Benchmark', 'NumFCNodes_MLPNN'))
		num_fc_layers = int(config.get('Benchmark', 'NumFCLayers_MLPNN'))
		lamb = float(config.get('Benchmark', 'L2_Lambda_MLPNN'))
		drop = float(config.get('Benchmark', 'Dropout_MLPNN'))

		reg = tf.keras.regularizers.l2(lamb)
		self.model = tf.keras.Sequential()

		self.model.add(tf.keras.layers.GaussianNoise(0.01, input_shape=(input_len,)))

		for i in range(0, num_fc_layers):
			self.model.add(tf.keras.layers.Dense(num_fc_nodes, activation='relu', kernel_regularizer=reg, bias_regularizer=reg, name="fc_"+str(i)))
			self.model.add(tf.keras.layers.Dropout(drop))

		self.model.add(tf.keras.layers.Dense(num_class, activation='softmax', kernel_regularizer=reg, bias_regularizer=reg, name="output"))

		self.patience = int(config.get('Benchmark', 'Patience_MLPNN'))
		self.learning_rate = float(config.get('Benchmark', 'LearningRate_MLPNN'))
		self.batch_size = int(config.get('Benchmark', 'BatchSize_MLPNN'))


	def train(self, train, train_weights=[]):
		train_x, train_y = train

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

		preds = self.model.predict(test_x)
		stats = get_stat_dict(test_y, preds)
		return preds, stats	


	def destroy(self):
		tf.keras.backend.clear_session()
		return