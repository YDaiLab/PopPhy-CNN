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
from popphy_io import load_data_from_file
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from  sklearn import preprocessing

tf.flags.DEFINE_integer("epochs", 500, "Number of epochs")
tf.flags.DEFINE_float("learning_rate", 0.002, "Learning rate.")
tf.flags.DEFINE_integer("batch_size", 32, "Minibatch size.")
tf.flags.DEFINE_integer("num_layers", 3, "Number of convolutional layers")
tf.flags.DEFINE_string("data_set", "Cirrhosis", "Dataset")
tf.flags.DEFINE_string("model", None, "Model directory")
tf.flags.DEFINE_string("activation", "relu", "Activation function")
tf.flags.DEFINE_string("stat", "MCC", "Stat to optimize on")
tf.flags.DEFINE_string("feature_set", "raw", "Feature Set")
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

def PopPhyCNN(num_features, num_class):

	X = tf.placeholder("float", [None,  1, num_features, 1], name="X")
	Y = tf.placeholder("float", [None], name="Y")
        CW = tf.placeholder("float", [None], name="CW")

	lab = tf.cast(Y, dtype=tf.int32)
	labels_oh = tf.one_hot(lab, depth=num_class)	

	regularizer = tf.contrib.layers.l2_regularizer(scale=0.2)
	
	with tf.variable_scope("CNN", reuse=False) as vs:
	    conv1 = tf.layers.conv2d(X, 64, (1,10), activation=tf.nn.relu, kernel_regularizer=regularizer, bias_regularizer=regularizer)	
	    conv1 = tf.layers.max_pooling2d(conv1, (1,2), (1,2))

	    conv2 = tf.layers.conv2d(conv1, 64, (1,10), activation=tf.nn.relu, kernel_regularizer=regularizer, bias_regularizer=regularizer)	
	    conv2 = tf.layers.max_pooling2d(conv2, (1,2), (1,2))
	    
	    conv3 = tf.layers.conv2d(conv2, 64, (1,10), activation=tf.nn.relu, kernel_regularizer=regularizer, bias_regularizer=regularizer)	
	    conv3 = tf.layers.max_pooling2d(conv3, (1,2), (1,2))

	with tf.variable_scope("Output"):
	    conv_flat = tf.contrib.layers.flatten(conv3)
	    fc1 = tf.layers.dense(conv_flat, 128, activation=tf.nn.relu, kernel_regularizer=regularizer, bias_regularizer=regularizer)
	    fc1_drop = tf.layers.dropout(fc1, rate=0.2)
	    fc2 = tf.layers.dense(fc1_drop, 128, activation=tf.nn.relu, kernel_regularizer=regularizer, bias_regularizer=regularizer)        
	    fc2_drop = tf.layers.dropout(fc2, rate=0.2)
            out = tf.layers.dense(fc2_drop, num_class, kernel_regularizer=regularizer, bias_regularizer=regularizer)

	    prob = tf.nn.softmax(out, axis=1)

        with tf.name_scope("Loss"):
                losses = tf.losses.softmax_cross_entropy(onehot_labels=labels_oh, logits=out)
                mean_loss = tf.reduce_mean(losses, name="mean_loss")
	
        return {'cost': mean_loss, 'x':X, 'y':Y, 'cw':CW, 'prob':prob[:,1], 'pred':prob}	

def train():

    num_set = 10
    num_cv = 10
    method = "CV"
    auc_list = []
    mcc_list = []
    for set in range(0, num_set):
        for cv in range(0,num_cv):
            print("\nRun " + str(cv) + " of set " + str(set) + ".\n")
            

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
	    print("There are %d features" % (num_features))
            train = x.as_matrix().reshape(x.shape[0], 1, num_features, 1)
            test = tx.as_matrix().reshape(tx.shape[0], 1, num_features, 1)

            c_prob = [None] * len(np.unique(y))
            for l in np.unique(y):
                c_prob[int(l)] = float( float(len(y))/ (2.0 * float((np.sum(y == l)))))

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

 
	    train_weights = np.array(train_weights)
	    test_weights = np.array(test_weights)

            best_roc = 0
	    best_mcc = -1
            best_pred_l = []
            best_y= []
            patience = 20
            for run in range(0,1):
	    	print("Building model...")
		run_roc = 0
		run_mcc = -1

                train_dataset = tf.data.Dataset.from_tensor_slices((train, y, train_weights))
                train_dataset = train_dataset.batch(batch_size)

                test_dataset = tf.data.Dataset.from_tensor_slices((test, ty, test_weights))
                test_dataset = test_dataset.batch(batch_size)

	        train_iterator = train_dataset.make_initializable_iterator()
	        next_train_element = train_iterator.get_next()

           	test_iterator = test_dataset.make_initializable_iterator()
        	next_test_element = test_iterator.get_next()

	    	model = PopPhyCNN(num_features, num_class)
	
		with tf.name_scope("Train"):
			optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model['cost'])

		init = tf.global_variables_initializer()

		print("Starting session...")
	   	with tf.device("/gpu:0"):
			sess = tf.Session()
			sess.run(init)
	
            		best_loss = -1
			#merged_summary = tf.summary.merge_all()
			#writer = tf.summary.FileWriter(LOG_DIR)
			#writer.add_graph(sess.graph)
			i = 0	
			while True:
				i += 1	
				#print("Starting Epoch %d" % (i))
				training_loss = 0
				validation_loss = 0
            			num_training_batches = 0
            			num_test_batches =  0
				y_list = []
				out_list = []
				pred_lab_list = []
				sess.run(train_iterator.initializer)
				while True:
					try:
						batch_x, batch_y, batch_cw = sess.run(next_train_element)
						_, l = sess.run([optimizer, model['cost']], feed_dict={model['x']: batch_x, model['y']: batch_y.reshape(-1), model['cw']:batch_cw})
						training_loss += l
						num_training_batches += 1
					except tf.errors.OutOfRangeError:
						#print('Epoch %i Loss: %f' % (i, training_loss/num_training_batches))
						break
				
				sess.run(test_iterator.initializer)
                                
                                while True:
                                        try:
                                                batch_x, batch_y, batch_cw = sess.run(next_test_element)
                                                l, pred, out, y_out = sess.run([model['cost'], model['prob'], model['pred'], model['y']], feed_dict={model['x']: batch_x, model['y']: batch_y.reshape(-1), model['cw']:batch_cw})
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
                                                                #save_path = saver.save(sess, dir + "/model_"+str(num_layers)+"L_MCC.chkpt")
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
                                                                #save_path = saver.save(sess, dir + "/model_"+str(num_layers)+"L_AUC.chkpt")

                                                break


				if patience == 0 or (stat =="MCC" and best_mcc == 1) or (stat=="AUC" and best_roc==1):
                                        #fm = sess.run([model['fm']], feed_dict={model['x']:train, model['y']:y.reshape(-1), model['cw']:train_weights, model['training']:False, model['tree']:tree_train})
                                        #f = open(dir+"/fm_" + str(num_layers) + "L_" + stat + ".save", 'wb')
                                        #cPickle.dump(fm, f, protocol=cPickle.HIGHEST_PROTOCOL)
                                        #f.close()



                                        print("Early Stopping at Epoch %i" % (i))
                                        print("Run AUC: %f" % (run_roc))
                                        print("Best AUC: %f" % (best_roc))
                                        print("Run MCC: %f" % (run_mcc))
                                        print("Best MCC: %f" % (best_mcc))
                                        tf.reset_default_graph()
                                        del(sess)
                                        break

            auc_list.append(best_roc)
            mcc_list.append(best_mcc)
            print("Current AUC average: %f" % (np.mean(np.array(auc_list))))
            print("Current MCC average: %f" % (np.mean(np.array(mcc_list))))
    print("------------------------------------------------------------------")
    print("Mean AUC: %f" % (np.mean(np.array(auc_list))))
    print("Mean MCC: %f" % (np.mean(np.array(mcc_list))))
    print("1D CNN " + dset)

    dir = "../data/" + dset + "/data_sets/"
    f = open(dir + "results_1d_" + feature_set + "_" + stat +".txt", 'w')
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
	
