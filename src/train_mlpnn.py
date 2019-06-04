from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data
from math import log, sqrt
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from popphy_io import load_data_from_file
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from  sklearn import preprocessing
from sklearn.model_selection import train_test_split

np.seterr(all='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.flags.DEFINE_integer("epochs", 500, "Number of epochs")
tf.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate.")
tf.flags.DEFINE_integer("batch_size", 32, "Minibatch size.")
tf.flags.DEFINE_integer("num_layers", 3, "Number of convolutional layers")
tf.flags.DEFINE_string("data_set", "Cirrhosis", "Dataset")
tf.flags.DEFINE_string("model", None, "Model directory")
tf.flags.DEFINE_string("activation", "relu", "Activation function")
tf.flags.DEFINE_string("stat", "AUC", "Stat to optimize on")
tf.flags.DEFINE_string("feature_set", "raw", "Feature set")
tf.flags.DEFINE_integer("num_class", 2, "Number of classes")
FLAGS = tf.flags.FLAGS

LOG_DIR= os.getcwd() + "/logs/" + str(FLAGS.model)

# Training Parameters
learning_rate = FLAGS.learning_rate
batch_size = FLAGS.batch_size
act_fn = FLAGS.activation
num_layers = FLAGS.num_layers
dset = FLAGS.data_set
stat = FLAGS.stat
feature_set = FLAGS.feature_set
num_class = FLAGS.num_class

def PopPhyMLPNN(num_features, num_class):

	X = tf.placeholder("float", [None,  num_features], name="X")
	Y = tf.placeholder("float", [None], name="Y")
	CW = tf.placeholder("float", [None], name="CW")
	Training = tf.placeholder(tf.bool)

	lab = tf.cast(Y, dtype=tf.int32)
	labels_oh = tf.one_hot(lab, depth=num_class)	
	regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
	
	with tf.variable_scope("Output"):
		fc = tf.layers.dense(X, 32, activation=tf.nn.relu, kernel_regularizer=regularizer, bias_regularizer=regularizer)
		fc_drop = tf.layers.dropout(fc, rate=0.5, training=Training)
		fc1 = tf.layers.dense(fc_drop, 32, activation=tf.nn.relu, kernel_regularizer=regularizer, bias_regularizer=regularizer)
		fc1_drop = tf.layers.dropout(fc1, rate=0.5, training=Training)
		out = tf.layers.dense(fc1_drop, num_class, kernel_regularizer=regularizer, bias_regularizer=regularizer)

		prob = tf.nn.softmax(out, axis=1)

	with tf.name_scope("Loss"):
		losses = tf.losses.softmax_cross_entropy(onehot_labels=labels_oh, logits=out)
		losses = losses + tf.losses.get_regularization_loss()
		mean_loss = tf.reduce_mean(losses, name="mean_loss")
	
	return {'cost': mean_loss, 'x':X, 'y':Y, 'cw':CW, 'prob':prob[:,1], 'pred':prob, 'training':Training}	

def train():

	num_set = 10
	num_cv = 10
	method = "CV"
	auc_list = []
	mcc_list = []
	for set in range(0, num_set):
		for cv in range(0,num_cv):
			print("\nRun " + str(cv) + " of set " + str(set) + ".\n")
		
			train, test, validation = load_data_from_file(dset, method, str(set), str(cv))
			dir = "../data/" + dset + "/data_sets/" + method + "_" + str(set) + "/" + str(cv)
			if feature_set == "raw":
				x = pd.read_csv(dir+"/benchmark_train_data.csv", header=None, dtype=np.float64)
				tx = pd.read_csv(dir+"/benchmark_test_data.csv", header=None, dtype=np.float64)
			elif feature_set == "tree":
				x = pd.read_csv(dir+"/benchmark_tree_train_data.csv", header=None, dtype=np.float64)
				tx = pd.read_csv(dir+"/benchmark_tree_test_data.csv", header=None, dtype=np.float64)

			y = np.asarray(np.loadtxt(dir+'/benchmark_train_labels.csv', delimiter=','))
			ty = np.asarray(np.loadtxt(dir+'/benchmark_test_labels.csv', delimiter=','))

			train = np.asarray(x)
			test = np.asarray(tx)
		
			num_features = train.shape[1]
			train = x.values.reshape(-1, num_features)
			test = tx.values.reshape(-1, num_features)


			c_prob = [None] * len(np.unique(y))
			for l in np.unique(y):
				c_prob[int(l)] = float( float(len(y))/ (2.0 * float((np.sum(y == l)))))


			if method == "CV":
				dir = "../data/" + dset + "/data_sets/CV_"  + str(set) + "/" + str(cv)
			if method == "HO" or method == "holdout":
				dir = "../data/" + dset + "/data_sets/HO_"  + str(set)

 
			train_x, valid_x, train_y, valid_y = train_test_split(train, y, test_size=0.1)
	
			test_x = test
			scaler = MinMaxScaler().fit(train_x)
			train_x = np.clip(scaler.transform(train_x),0,1)
			valid_x = np.clip(scaler.transform(valid_x),0,1)
			test_x = np.clip(scaler.transform(test),0,1)

			train_weights = []
			valid_weights = []
			test_weights = []

			for i in train_y:
				train_weights.append(c_prob[int(i)])
			for i in valid_y:
				valid_weights.append(c_prob[int(i)])
			for i in ty:
				test_weights.append(c_prob[int(i)])

			train_weights = np.array(train_weights)
			valid_weights = np.array(valid_weights)
			test_weights = np.array(test_weights)

			patience = 200
			fm = None
			pred = None
			w = None
			b = None
			eval_train = None
			eval_y = None
	
			train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y, train_weights))
			train_dataset = train_dataset.batch(batch_size)
	
			valid_dataset = tf.data.Dataset.from_tensor_slices((valid_x, valid_y, valid_weights))
			valid_dataset = train_dataset.batch(batch_size)
	
			test_dataset = tf.data.Dataset.from_tensor_slices((test, ty, test_weights))
			test_dataset = test_dataset.batch(batch_size)
	
			train_iterator = train_dataset.make_initializable_iterator()
			next_train_element = train_iterator.get_next()
	
			valid_iterator = train_dataset.make_initializable_iterator()
			next_valid_element = valid_iterator.get_next()
	
			test_iterator = test_dataset.make_initializable_iterator()
			next_test_element = test_iterator.get_next()
	
			model = PopPhyMLPNN(num_features, num_class)
	
			with tf.name_scope("Train"):
				optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model['cost'])
	
			init = tf.global_variables_initializer()
			saver = tf.train.Saver()
			print("Starting session...")
			with tf.Session() as sess:
				sess.run(init)
				p=patience
				val_roc = 0
				val_mcc = -1
				best_val = -1
				roc = 0
				mcc = 0
				i = 0
				trained_epochs=0
				while p > 0:
					i += 1
					p -= 1
					training_loss = 0
					validation_loss = 0
					num_training_batches = 0
					num_valid_batches = 0
					num_test_batches =  0
					y_list = []
					pred_lab_list = []
					out_list = []

					sess.run(train_iterator.initializer)
					while True:
						try:
							batch_x, batch_y, batch_cw = sess.run(next_train_element)
							_, l = sess.run([optimizer, model['cost']], feed_dict={model['x']: batch_x,
								model['y']: batch_y.reshape(-1), model['cw']:batch_cw, model['training']:True})
							training_loss += l
							num_training_batches += 1
						except tf.errors.OutOfRangeError:
							break

					sess.run(valid_iterator.initializer)
					y_list = []
					pred_lab_list = []
					while True:
						try:
							batch_x,  batch_y, batch_cw = sess.run(next_valid_element)
							l, pred, out, y_out = sess.run([model['cost'], model['prob'], model['pred'], model['y']],
								feed_dict={model['x']: test_x, model['y']: ty.reshape(-1), model['cw']:test_weights,
								model['training']:False})
							validation_loss += l
							num_test_batches += 1
							y_list = list(y_list) + list(y_out)
							pred_lab_list = list(pred_lab_list) + list(pred)
							for row in out:
								out_list.append(np.argmax(row))
						except tf.errors.OutOfRangeError:
							loss = validation_loss/float(num_test_batches)
							y_list = np.array(y_list).reshape((-1,1))
							pred_lab_list = np.array(pred_lab_list).reshape((-1,1))
							pred_lab_list_binary = np.where(pred_lab_list > 0.5, 1, 0)
							roc = roc_auc_score(y_list, pred_lab_list)
							mcc = matthews_corrcoef(y_list, pred_lab_list_binary)
							if stat == "AUC" and roc > best_val:
								p = patience
								trained_epochs = i
								best_val = roc
								save_path = saver.save(sess, dir+"/MLPNN.ckpt")	
							if stat == "MCC" and mcc > best_val:
								p = patience
								trained_epochs = i
								best_val = mcc
								save_path = saver.save(sess, dir+"/MLPNN.ckpt")	
							break
	
				saver.restore(sess, dir+"/MLPNN.ckpt")
				for j in range(0,np.clip(int(trained_epochs/10.0),5,100)):
					sess.run(valid_iterator.initializer)
					while True:
						try:
							batch_x, batch_y, batch_cw = sess.run(next_valid_element)
							_ = sess.run(optimizer, feed_dict={model['x']: batch_x,
								model['y']: batch_y.reshape(-1), model['cw']:batch_cw, model['training']:True})
						except tf.errors.OutOfRangeError:
							break
				save_path = saver.save(sess, dir+"/MLPNN.ckpt")	
				y_list = []
				pred_lab_list = []
				sess.run(test_iterator.initializer)
				while True:
						try:
							batch_x,  batch_y, batch_cw = sess.run(next_test_element)
							l, pred, out, y_out = sess.run([model['cost'], model['prob'], model['pred'], model['y']],
								feed_dict={model['x']: test_x, model['y']: ty.reshape(-1), model['cw']:test_weights,
								model['training']:False})
							validation_loss += l
							num_test_batches += 1
							y_list = list(y_list) + list(y_out)
							pred_lab_list = list(pred_lab_list) + list(pred)

						except tf.errors.OutOfRangeError:
							y_list = np.array(y_list).reshape((-1,1))
							pred_lab_list = np.array(pred_lab_list).reshape((-1,1))
							pred_lab_list_binary = np.where(pred_lab_list > 0.5, 1, 0)
							roc = roc_auc_score(y_list, pred_lab_list)
							mcc = matthews_corrcoef(y_list, pred_lab_list_binary)
							break


			tf.reset_default_graph()	
			auc_list.append(roc)
			mcc_list.append(mcc)
			print("Current AUC average: %f" % (np.mean(np.array(auc_list))))
			print("Current MCC average: %f" % (np.mean(np.array(mcc_list))))
	print("------------------------------------------------------------------")
	print("Mean AUC: %f" % (np.mean(np.array(auc_list))))
	print("Mean MCC: %f" % (np.mean(np.array(mcc_list))))
	print("MLPNN " + dset)

	dir = "../data/" + dset + "/data_sets/"
	f = open(dir + "results_mlpnn_" + feature_set + "_" + stat +".txt", 'w')
	f.write("\nMean ROC: " + str(np.mean(auc_list)) + " (" + str(np.std(auc_list)) + ")\n")
	f.write(str(auc_list) + "\n")
	f.write("\nMean MCC: " + str(np.mean(mcc_list)) + " (" + str(np.std(mcc_list)) + ")\n")
	f.write(str(mcc_list) + "\n")
	f.close()

	return

def main(_):
	global num_input

	if FLAGS.model == None:
		print("Please specify model directory.")
	
	if not os.path.exists(LOG_DIR):
		os.makedirs(LOG_DIR)
	train()		
	
if __name__ == '__main__':
	tf.app.run(main=main)
	
