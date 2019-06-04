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
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from feature_map_analysis import get_feature_map_rankings
import time
import copy

np.seterr(all='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.flags.DEFINE_integer("epochs", 500, "Number of epochs")
tf.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate.")
tf.flags.DEFINE_integer("batch_size", 512, "Minibatch size.")
tf.flags.DEFINE_integer("num_kernels", 32, "Number of convolutional kernels.")
tf.flags.DEFINE_integer("num_layers", 3, "Number of convolutional layers")
tf.flags.DEFINE_integer("num_sets", 10, "Number of sets")
tf.flags.DEFINE_integer("num_cv", 10, "Number of CV partitions")
tf.flags.DEFINE_string("data_set", "Cirrhosis", "Dataset")
tf.flags.DEFINE_string("model", None, "Model directory")
tf.flags.DEFINE_string("activation", "relu", "Activation function")
tf.flags.DEFINE_string("stat", "AUC", "Stat to optimize on")
tf.flags.DEFINE_bool("out", True, "Write to output")
tf.flags.DEFINE_integer("num_class", 2, "Number of classes")
tf.flags.DEFINE_bool("eval_features", False, "Evaluate Features")
tf.flags.DEFINE_float("thresh", 0.6, "Feature extraction threshold")
tf.flags.DEFINE_float("theta_1", 0.1, "Feature extraction threta 1")
tf.flags.DEFINE_float("theta_2", 0.25, "Feature extraction threta 2")
FLAGS = tf.flags.FLAGS

LOG_DIR= os.getcwd() + "/logs/" + str(FLAGS.model)

# Training Parameters
learning_rate = FLAGS.learning_rate
batch_size = FLAGS.batch_size
act_fn = FLAGS.activation
num_layers = FLAGS.num_layers
dset = FLAGS.data_set
stat = FLAGS.stat
to_write = FLAGS.out
thresh = FLAGS.thresh
num_class = FLAGS.num_class
num_kernels = FLAGS.num_kernels
eval_features = FLAGS.eval_features
num_sets = FLAGS.num_sets
num_cv = FLAGS.num_cv
theta_1 = FLAGS.theta_1
theta_2 = FLAGS.theta_2

def PopPhyCNN(num_row, num_col, num_features, num_class, lamb=0.1, drop=0.1):

	X = tf.placeholder("float", [None,  num_row, num_col, 1], name="X")
	Y = tf.placeholder("float", [None], name="Y")
	Noise = tf.placeholder("float", [None, num_row, num_col, 1], name="Noise")
	CW = tf.placeholder("float", [None], name="CW")
	training = tf.placeholder(tf.bool)
	batch_size = tf.placeholder(tf.int32)
	lab = tf.cast(Y, dtype=tf.int32)
	labels_oh = tf.one_hot(lab, depth=num_class)

	regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)

	with tf.variable_scope("CNN", reuse=False):

		#X = tf.add(X, Noise)
		conv = tf.layers.conv2d(X, num_kernels, (5, 5), name="Conv", activation=None,
			kernel_regularizer=regularizer, bias_regularizer=regularizer, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
		conv = tf.nn.relu(conv)
		conv_pool = tf.layers.max_pooling2d(conv, (2,2), (2,2))
		fm = conv
		if num_layers == 2:
			conv = tf.layers.conv2d(conv_pool, num_kernels, (4, 4), name="Conv2", activation=None,
				kernel_regularizer=regularizer, bias_regularizer=regularizer, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
			conv = tf.nn.relu(conv)
			conv_pool = tf.layers.max_pooling2d(conv, (2,2), (2,2))

		if num_layers == 3:
			conv = tf.layers.conv2d(conv_pool, num_kernels, (3, 3), name="Conv3", activation=None,
				kernel_regularizer=regularizer, bias_regularizer=regularizer, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
			conv = tf.nn.relu(conv)
			conv_pool = tf.layers.max_pooling2d(conv, (1,2), (1,2))

	with tf.variable_scope("FC"):
		fc = tf.contrib.layers.flatten(conv_pool)

		fc = tf.layers.dense(fc, 32, activation=tf.nn.relu, kernel_regularizer=regularizer, bias_regularizer=regularizer, name="Layer1")
		fc = tf.layers.dropout(fc, rate=0.5, training=training)

	with tf.variable_scope("Output"):
		out = tf.layers.dense(fc, num_class, kernel_regularizer=regularizer, bias_regularizer=regularizer, name="Out")

	prob = tf.nn.softmax(out, axis=1)

	with tf.name_scope("Loss"):
		ce_loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_oh, logits=out, weights=CW)
		losses = ce_loss + tf.losses.get_regularization_loss()
		cost = tf.reduce_sum(losses)

	return {'ce_cost': ce_loss, 'cost': cost, 'x':X, 'y':Y, 'cw':CW, 'prob':prob[:,1], 'pred':prob, 'training':training, 'noise':Noise, 'batch_size':batch_size, 'fm':fm}

def train():
	col_index = 0
	method = "CV"
	auc_list = []
	mcc_list = []
	rank_df = {}
	score_df = {}
	start = time.time()
	for set in range(0, num_sets):
		for cv in range(0,num_cv):
			print("\nRun " + str(cv) + " of set " + str(set) + ".\n")
			train, test, validation = load_data_from_file(dset, method, str(set), str(cv))
			dir = "../data/" + str(dset) + "/data_sets/CV_" + str(set) + "/" + str(cv)
			y = np.loadtxt(dir + '/benchmark_train_labels.csv', delimiter=',')
			ty = np.loadtxt(dir+'/benchmark_test_labels.csv', delimiter=',')
			c_prob = [None] * len(np.unique(y))
			for l in np.unique(y):
				c_prob[int(l)] = float( float(len(y))/ (2.0 * float((np.sum(y == l)))))

			tree_x = pd.read_csv(dir+"/benchmark_tree_train_data.csv", header=None, dtype=np.float64)
			tree_tx = pd.read_csv(dir+"/benchmark_tree_test_data.csv", header=None, dtype=np.float64)

			tree_train = np.asarray(tree_x)
			tree_test = np.asarray(tree_tx)

			num_features = tree_train.shape[1]
			labels = y

		

			if method == "CV":
				dir = "../data/" + dset + "/data_sets/CV_"  + str(set) + "/" + str(cv)
			if method == "HO" or method == "holdout":
				dir = "../data/" + dset + "/data_sets/HO_"  + str(set)

			rows = train.shape[1]
			cols = train.shape[2]

			train = np.array(train).reshape((-1,rows*cols))
			test = np.array(test).reshape((-1,rows*cols))
			train_x, valid_x, train_tree, valid_tree, train_y, valid_y = train_test_split(train, tree_train,  y, test_size=0.1)
		
			test_x = test
			scaler = MinMaxScaler().fit(train)
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
			train = np.array(train).reshape((-1, rows, cols,1))
			train_x = np.array(train_x).reshape((-1,rows,cols,1))
			valid_x = np.array(valid_x).reshape((-1,rows,cols,1))
			test_x = np.array(test_x).reshape((-1,rows,cols,1))
	
			train_weights = np.array(train_weights)
			test_weights = np.array(test_weights)

			patience = 200
			fm = None
			pred = None
			w = None
			b = None
			eval_train = None
			eval_y = None

			train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_tree, train_y, train_weights))
			train_dataset = train_dataset.batch(batch_size)

			valid_dataset = tf.data.Dataset.from_tensor_slices((valid_x, valid_tree, valid_y, valid_weights))
			valid_dataset = train_dataset.batch(batch_size)
		
			test_dataset = tf.data.Dataset.from_tensor_slices((test, tree_test,ty, test_weights))
			test_dataset = test_dataset.batch(batch_size)

			train_iterator = train_dataset.make_initializable_iterator()
			next_train_element = train_iterator.get_next()

			valid_iterator = train_dataset.make_initializable_iterator()
			next_valid_element = valid_iterator.get_next()

			test_iterator = test_dataset.make_initializable_iterator()
			next_test_element = test_iterator.get_next()

			model = PopPhyCNN(rows, cols, num_features, num_class)
		
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
				best_model = None
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
							batch_x, batch_tree, batch_y, batch_cw = sess.run(next_train_element)
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
							batch_x, batch_tree, batch_y, batch_cw = sess.run(next_valid_element)
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
								save_path = saver.save(sess, dir+"/PopPhy-CNN.ckpt")	
							if stat == "MCC" and mcc > best_val:
								p = patience
								trained_epochs = i
								best_val = mcc
								save_path = saver.save(sess, dir+"/PopPhy-CNN.ckpt")	
							break
				saver.restore(sess, dir+"/PopPhy-CNN.ckpt")
				for j in range(0,np.clip(int(trained_epochs/10.0),1,100)):
					sess.run(valid_iterator.initializer)
					while True:
						try:
							batch_x, batch_tree, batch_y, batch_cw = sess.run(next_valid_element)
							_ = sess.run(optimizer, feed_dict={model['x']: batch_x, 
								model['y']: batch_y.reshape(-1), model['cw']:batch_cw, model['training']:True})
						except tf.errors.OutOfRangeError:
							break
				save_path = saver.save(sess, dir+"/PopPhy-CNN.ckpt")	
				y_list = []
				pred_lab_list = []
				sess.run(test_iterator.initializer)
				while True:
						try:
							batch_x, batch_tree, batch_y, batch_cw = sess.run(next_test_element)
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
							if eval_features and roc > thresh:
								print("Starting Feature Evaluation")
								fm, pred = sess.run([model['fm'], model['prob']], feed_dict={model['x']:train_x, 
									model['y']:train_y, model['cw']:train_weights, model['training']:False})
								eval_l = np.where(pred > 0.5, 1, 0)
								with tf.variable_scope('CNN', reuse=tf.AUTO_REUSE):
									w = tf.get_variable('Conv/kernel').eval()
									b = tf.get_variable('Conv/bias').eval()
								eval_train = train_x
								eval_y = train_y
							break
			tf.reset_default_graph()
			if eval_features and roc > thresh:
				print("Getting Feature Rankings")
				scores, rankings = get_feature_map_rankings(dset, eval_train, eval_y, eval_l, fm, w, b, theta_1, theta_2)
				
				if col_index == 0:
					for l in np.unique(y):
						score_df[l] =  scores[l]
						rank_df[l] = rankings[l]
	
				else:
					for l in np.unique(y):
						score_df[l] = score_df[l].join(scores[l], rsuffix = str(col_index))
						rank_df[l] = rank_df[l].join(rankings[l], rsuffix = str(col_index))
				col_index += 1
			print("Finished")
			auc_list.append(roc)
			mcc_list.append(mcc)
			print("Current AUC average: %f" % (np.mean(np.array(auc_list))))
			print("Current MCC average: %f" % (np.mean(np.array(mcc_list))))
	print("------------------------------------------------------------------")
	print("Mean AUC: %f" % (np.mean(np.array(auc_list))))
	print("Mean MCC: %f" % (np.mean(np.array(mcc_list))))
	print("PopPhy " + dset)


	if to_write:
		dir = "../results/"
	if not os.path.exists(dir):
		os.makedirs(dir)
	f = open(dir + dset + "_PopPhy_" + str(num_layers) + "L_" + stat +".txt", 'w')
	f.write("\nMean ROC: " + str(np.mean(auc_list)) + " (" + str(np.std(auc_list)) + ")\n")
	f.write(str(auc_list) + "\n")
	f.write("\nMean MCC: " + str(np.mean(mcc_list)) + " (" + str(np.std(mcc_list)) + ")\n")
	f.write(str(mcc_list) + "\n")
	f.close()

	if eval_features:
		fp = open("../data/" + dset +"/label_reference.txt", 'r')
		labels = fp.readline().split("['")[1].split("']")[0].split("' '")
		fp.close()

		for i in range(0, len(labels)):
			lab = labels[i]
			dir = "../data/" + dset + "/" + labels[i]
			quantiles = rank_df[i].quantile(q=0.25, axis=1)
			max_score = score_df[i].max(axis=1)
			medians = rank_df[i].median(axis=1)
			mins = rank_df[i].min(axis=1)
			max_rankings = max_score.rank(ascending=False) 
			medians.to_csv(dir + "_median_rankings.csv", header=None)
			quantiles.to_csv(dir + "_quantile_rankings.csv", header=None)
			max_rankings.to_csv(dir + "_max_rankings.csv", header=None)
			mins.to_csv(dir + "_min_rankings.csv", header=None)
			rank_df[i].to_csv(dir + "_rankings.csv")
			score_df[i].to_csv(dir + "_scores.csv")

	return

def main(_):
	global num_input
	train()		
	
if __name__ == '__main__':
	import warnings
	warnings.filterwarnings("ignore", category=DeprecationWarning)
	tf.app.run(main=main)
	
