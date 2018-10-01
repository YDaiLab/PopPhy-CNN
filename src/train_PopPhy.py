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
from sklearn import preprocessing
from feature_map_analysis import get_feature_map_rankings
import cPickle
import time

tf.flags.DEFINE_integer("epochs", 500, "Number of epochs")
tf.flags.DEFINE_float("learning_rate", 0.002, "Learning rate.")
tf.flags.DEFINE_integer("batch_size", 32, "Minibatch size.")
tf.flags.DEFINE_integer("num_kernels", 64, "Minibatch size.")
tf.flags.DEFINE_integer("num_layers", 3, "Number of convolutional layers")
tf.flags.DEFINE_integer("num_sets", 10, "Number of sets")
tf.flags.DEFINE_integer("num_cv", 10, "Number of CV partitions")
tf.flags.DEFINE_string("data_set", "Cirrhosis", "Dataset")
tf.flags.DEFINE_string("model", None, "Model directory")
tf.flags.DEFINE_string("activation", "relu", "Activation function")
tf.flags.DEFINE_string("stat", "MCC", "Stat to optimize on")
tf.flags.DEFINE_bool("out", True, "Write to output")
tf.flags.DEFINE_integer("num_class", 2, "Number of classes")
tf.flags.DEFINE_bool("eval_features", False, "Evaluate Features")
tf.flags.DEFINE_float("thresh", 0.5, "Feature extraction threshold")
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

def PopPhyCNN(num_row, num_col, num_features, num_class):

	X = tf.placeholder("float", [None,  num_row, num_col, 1], name="X")
	Y = tf.placeholder("float", [None], name="Y")
        CW = tf.placeholder("float", [None], name="CW")
	training = tf.placeholder(tf.bool)
	tree = tf.placeholder("float", [None, num_features], name="Tree")
	lab = tf.cast(Y, dtype=tf.int32)
	labels_oh = tf.one_hot(lab, depth=num_class)	

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.2)
	
	with tf.variable_scope("CNN", reuse=False) as vs:
	
	    conv = tf.layers.conv2d(X, num_kernels, (5,10), name="conv1", activation=tf.nn.relu, kernel_regularizer=regularizer, bias_regularizer=regularizer, use_bias=True)	
	    fm = conv
	    conv = tf.layers.max_pooling2d(conv, (2,2), (2,2))
	   
	    for l in range(num_layers - 1):
	    	conv = tf.layers.conv2d(conv, num_kernels, (4-l,10), activation=tf.nn.relu, kernel_regularizer=regularizer, bias_regularizer=regularizer, use_bias=True)	
		conv = tf.layers.max_pooling2d(conv, (min(conv.shape[1],2),2), (min(conv.shape[1],2),2))
	    

	with tf.variable_scope("Output"):
	    conv_flat = tf.contrib.layers.flatten(conv)
	    fc1 = tf.layers.dense(conv_flat, 128, activation=tf.nn.relu, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	    fc1_drop = tf.layers.dropout(fc1, rate=0.2)
	    fc2 = tf.layers.dense(fc1_drop, 128, activation=tf.nn.relu, kernel_regularizer=regularizer, bias_regularizer=regularizer)       
	    fc2_drop = tf.layers.dropout(fc2, rate=0.2)
            out = tf.layers.dense(fc2_drop, num_class, kernel_regularizer=regularizer, bias_regularizer=regularizer)

	    prob = tf.nn.softmax(out, axis=1)

        with tf.name_scope("Loss"):
                losses = tf.losses.softmax_cross_entropy(onehot_labels=labels_oh, logits=out)
                mean_loss = tf.reduce_mean(losses, name="mean_loss")
	
	
	return {'cost': mean_loss, 'x':X, 'y':Y, 'cw':CW, 'prob':prob[:,1], 'pred':prob, 'training':training, 'tree':tree, 'fm':fm} 
	

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
	
            train_weights = []
            test_weights = []

            for i in y:
		train_weights.append(c_prob[int(i)])

	    for i in ty:
		test_weights.append(c_prob[int(i)])
		

            if method == "CV":
                dir = "../data/" + dset + "/data_sets/CV_"  + str(set) + "/" + str(cv)
            if method == "HO" or method == "holdout":
                dir = "../data/" + dset + "/data_sets/HO_"  + str(set)

            rows = train.shape[1]
            cols = train.shape[2]

	    train = np.array(train).reshape((-1,rows*cols))
	    test = np.array(test).reshape((-1,rows*cols))

	    train = np.array(train).reshape((-1,rows,cols,1))
	    test = np.array(test).reshape((-1,rows,cols,1))
	    valid = np.array(validation).reshape((-1,rows,cols,1))
	
	    train_weights = np.array(train_weights)
	    test_weights = np.array(test_weights)

            best_roc = 0
	    best_mcc = -1
            best_pred_l = []
            best_y= []
	    best_out = []
            patience = 40
	    fm = None
	    pred = None
	    w = None
	    b = None
	    eval_train = None
	    eval_y = None

            for run in range(0,1):
	    	print("Building model...")

                train_dataset = tf.data.Dataset.from_tensor_slices((train, tree_train, y, train_weights))
                train_dataset = train_dataset.batch(batch_size)

                test_dataset = tf.data.Dataset.from_tensor_slices((test, tree_test, ty, test_weights))
                test_dataset = test_dataset.batch(batch_size)

	        train_iterator = train_dataset.make_initializable_iterator()
	        next_train_element = train_iterator.get_next()

           	test_iterator = test_dataset.make_initializable_iterator()
        	next_test_element = test_iterator.get_next()

	    	model = PopPhyCNN(rows, cols, num_features, num_class)
	
		with tf.name_scope("Train"):
			optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model['cost'])

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		print("Starting session...")
	   	with tf.Session() as sess:
			#sess = tf.Session()
			sess.run(init)
				
            		best_loss = -1
			run_roc = 0
			run_mcc = -1
			i = 0	
			while True:
				i += 1	
				training_loss = 0
				validation_loss = 0
            			num_training_batches = 0
            			num_test_batches =  0
				y_list = []
				pred_lab_list = []
				out_list = []
				sess.run(train_iterator.initializer)
				while True:
					try:
						batch_x, batch_tree, batch_y, batch_cw = sess.run(next_train_element)
						_, l = sess.run([optimizer, model['cost']], feed_dict={model['x']: batch_x, model['y']: batch_y.reshape(-1), model['cw']:batch_cw, model['training']:True, model['tree']:batch_tree})
						training_loss += l
						num_training_batches += 1
					except tf.errors.OutOfRangeError:
						break
				
				sess.run(test_iterator.initializer)
				while True:
					try:
						batch_x, batch_tree, batch_y, batch_cw = sess.run(next_test_element)
						l, pred, out, y_out = sess.run([model['cost'], model['prob'], model['pred'], model['y']], feed_dict={model['x']: batch_x, model['y']: batch_y.reshape(-1), model['cw']:batch_cw, model['training']:False, model['tree']:batch_tree})
						validation_loss += l
						num_test_batches += 1
       	                                 	y_list = list(y_list) + list(y_out)
						pred_lab_list = list(pred_lab_list) + list(pred)
					        for row in out:
							out_list.append(np.argmax(row))
       	                         	except tf.errors.OutOfRangeError:
						patience -= 1
						loss = validation_loss/float(num_test_batches)
    						y_list = np.array(y_list).reshape((-1,1))
						pred_lab_list = np.array(pred_lab_list).reshape((-1,1))
						pred_lab_list_binary = np.where(pred_lab_list > 0.5, 1, 0)
						if num_class == 2:
							roc = roc_auc_score(y_list, pred_lab_list)
							mcc = matthews_corrcoef(y_list, pred_lab_list_binary)
						else:
							mcc = matthews_corrcoef(y_list, out_list)
							roc=0
						if best_loss == -1:
							best_loss = loss
					        if stat == "MCC":
                                                	if mcc > run_mcc or loss < best_loss:
                                                        	if mcc > run_mcc:
                                                               		run_mcc = mcc
								if roc > run_roc:
									run_roc = roc
                                                        	if mcc > best_mcc:
                                                                	best_roc = roc
                                                                	best_mcc = mcc
                                                        	if mcc == best_mcc and roc > best_roc:
                                                                	best_roc = roc
                                                        	if loss < best_loss:
                                                                	best_loss = loss
                                                        	patience = 40
								
								if eval_features:
                                                               		fm, pred = sess.run([model['fm'], model['prob']], feed_dict={model['x']:test, model['y']:ty.reshape(-1), model['cw']:train_weights, model['training']:False, model['tree']:tree_test})
                                                                	eval_l = np.where(pred > 0.5, 1, 0)
									with tf.variable_scope('CNN', reuse=True):
                                                                        	w = tf.get_variable('conv1/kernel').eval()
                                                                        	b = tf.get_variable('conv1/bias').eval()
									eval_train = test
									eval_y = ty
						if stat == "AUC":
                                                	if roc > run_roc or loss < best_loss:
                                                        	if roc > run_roc:
                                                                	run_roc = roc
								if mcc > run_mcc:
									run_mcc = mcc
                                                        	if roc > best_roc:
                                                                	best_roc = roc
                                                                	best_mcc = mcc
                                                        	if roc == best_roc and mcc > best_mcc:
                                                                	best_mcc = mcc
                                                        	if loss < best_loss:
                                                               		best_loss = loss
                                                        	patience = 40
								if eval_features:
									fm, pred = sess.run([model['fm'], model['prob']], feed_dict={model['x']:test, model['y']:ty.reshape(-1), model['cw']:train_weights, model['training']:False, model['tree']:tree_train})
                                                                	eval_l = np.where(pred > 0.5, 1, 0)
									with tf.variable_scope('CNN', reuse=True):
  										w = tf.get_variable('conv1/kernel').eval()
										b = tf.get_variable('conv1/bias').eval()
									eval_train = test
									eval_y = ty
							
						break

				if patience == 0 or (stat =="MCC" and best_mcc == 1) or (stat=="AUC" and best_roc==1):
                                        print("Early Stopping at Epoch %i" % (i))
                                        print("Run AUC: %f" % (run_roc))
                                        print("Best AUC: %f" % (best_roc))
                                        print("Run MCC: %f" % (run_mcc))
                                        print("Best MCC: %f" % (best_mcc))					

					if eval_features:
						print("Getting Feature Rankings")
						if (best_mcc >= thresh):
							scores, rankings = get_feature_map_rankings(dset, eval_train, eval_y, eval_l, fm, w, b)
						
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
					break

	    auc_list.append(best_roc)
	    mcc_list.append(best_mcc)
	    print("Current AUC average: %f" % (np.mean(np.array(auc_list))))
	    print("Current MCC average: %f" % (np.mean(np.array(mcc_list))))
	    tf.reset_default_graph()
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

	if FLAGS.model == None:
		print("Please specify model directory.")
	
	if not os.path.exists(LOG_DIR):
		os.makedirs(LOG_DIR)
	train()		
	
if __name__ == '__main__':
	tf.app.run(main=main)
	
