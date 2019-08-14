import sys
import os
from os.path import abspath
import numpy as np
import pandas as pd
from utils.generate_network import generate_network
from utils.prepare_data import prepare_data
from utils.popphy_io import get_config, save_params, load_params
from utils.popphy_io import get_stat, get_stat_dict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from models.PopPhy import PopPhyCNN
import warnings
from datetime import datetime
import webbrowser
import subprocess
import json

config = get_config()
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.nan)

def train_PopPhy():

	#####################################################################
	# Read in Config File
	#####################################################################
	config = get_config()
	filt_thresh = config.get('Evaluation', 'FilterThresh')
	dataset = config.get('Evaluation', 'DataSet')

	time_stamp = int(np.round(datetime.timestamp(datetime.now()), 0))
	ts = str(filt_thresh) + "_" + str(time_stamp)

	result_path = "../results/" + dataset + "/" + ts
	print("Saving results to %s" % (result_path))
	try:
		os.mkdir("../results/")
	except OSError:
		pass

	try:
		os.mkdir("../results/" + dataset)
	except OSError:
		pass

	try:
		os.mkdir(result_path)
	except OSError:
		print("Creation of the result subdirectory failed...")
		print("Exiting...")
		sys.exit()

	try:
		os.mkdir(result_path + "/prediction_evaluation")
	except OSError:
		print("Creation of the prediction evaluation subdirectory failed...")
		print("Exiting...")
		sys.exit()

	try:
		os.mkdir(result_path + "/feature_evaluation")
	except OSError:
		print("Creation of the feature evaluation subdirectory failed...")
		print("Exiting...")
		sys.exit()
	
	num_runs = int(config.get('Evaluation', 'NumberRuns'))
	num_test = int(config.get('Evaluation', 'NumberTestSplits'))


	#########################################################################
	# Read in data and generate tree maps
	#########################################################################
	print("\nStarting PopPhy-CNN on %s..." % (dataset))
	path = "../data/" + dataset

	my_maps, _, _, _, tree_features, labels, label_set, g, feature_df = prepare_data(path, config)

	num_class = len(np.unique(labels))
	if num_class == 2:
		metric = "AUC"
	else:
		metric = "MCC"

	seed = np.random.randint(100)
	np.random.seed(seed)
	np.random.shuffle(my_maps)
	np.random.seed(seed)
	np.random.shuffle(labels)

	n_values = np.max(labels) + 1
	labels_oh = np.eye(n_values)[labels]
		
	print("There are %d classes...%s" % (num_class, ", ".join(label_set)))
	#########################################################################
	# Determine which models are being trained
	#########################################################################

	cv_list = ["Run_" + str(x) + "_CV_" + str(y) for x in range(num_runs) for y in range(num_test)]

	stat_df = pd.DataFrame(index=["AUC", "MCC", "Precision", "Recall", "F1"], columns=cv_list)
		
	#########################################################################
	# Set up seeds for different runs
	#########################################################################

	feature_scores = {}

	for l in label_set:
		feature_scores[l] = pd.DataFrame(index=tree_features)

	seeds = np.random.randint(1000, size=num_runs)
	run = 0
	for seed in seeds:

		#####################################################################
		# Set up CV partitioning
		#####################################################################

		print("Starting CV")
		skf = StratifiedKFold(n_splits=num_test, shuffle=True, random_state=seed)
		fold = 0

		#####################################################################
		# Begin k-fold CV
		#####################################################################
		for train_index, test_index in skf.split(my_maps, labels):

			#################################################################
			# Select and format training and testing sets
			#################################################################
			train_x, test_x = my_maps[train_index,:,:], my_maps[test_index,:,:]
			train_y, test_y = labels_oh[train_index,:], labels_oh[test_index,:]
			train_x = np.log(train_x + 1)
			test_x = np.log(test_x + 1)
			c_prob = [0] * len(np.unique(labels))

			train_weights = []

			for l in np.unique(labels):
				a = float(len(labels))
				b = 2.0 * float((np.sum(labels==l)))
				c_prob[int(l)] = a/b
				
			c_prob = np.array(c_prob).reshape(-1)

			for l in np.argmax(train_y, 1):
				train_weights.append(c_prob[int(l)])
			train_weights = np.array(train_weights)
				
			num_train_samples = train_x.shape[0]
			num_test_samples = test_x.shape[0]
			tree_row = train_x.shape[1]
			tree_col = train_x.shape[2]

			scaler = MinMaxScaler().fit(train_x.reshape(-1, tree_row * tree_col))
			train_x = np.clip(scaler.transform(train_x.reshape(-1, tree_row * tree_col)), 0, 1).reshape(-1, tree_row, tree_col)
			test_x = np.clip(scaler.transform(test_x.reshape(-1, tree_row * tree_col)), 0, 1).reshape(-1, tree_row, tree_col)

			train = [train_x, train_y]
			test = [test_x, test_y]


			#################################################################
			# Triain PopPhy-CNN model using tree maps
			#################################################################

			popphy_model = PopPhyCNN((tree_row, tree_col), num_class, config)

			if fold + run == 0:
				print(popphy_model.model.summary())	
				print("\n\n Run\tFold\t%s" % (metric))

			popphy_model.train(train, train_weights)					
			preds, stats = popphy_model.test(test)
			if num_class == 2:
				stat_df.loc["AUC"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["AUC"]
			stat_df.loc["MCC"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["MCC"]
			stat_df.loc["Precision"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Precision"]
			stat_df.loc["Recall"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Recall"]
			stat_df.loc["F1"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["F1"]
			print("# %d\t%d\t%.3f" % (run, fold, stats[metric]))
	
			scores = popphy_model.get_feature_scores(train, g, label_set, tree_features, config)
			for l in range(len(label_set)):
				score_list = scores[:,l]
				lab = label_set[l]
				feature_scores[lab]["Run_" + str(run) + "_CV_" + str(fold)] = score_list

	
			popphy_model.destroy()
			fold += 1
		run += 1
		#####################################################################
		# Save metric dataframes as files
		#####################################################################
		
	print("\nAggregating evaluations...")
	stat_df.to_csv(result_path + "/prediction_evaluation/results_popphy.tsv", sep="\t")
	for l in label_set:
		feature_scores[l].to_csv(result_path + "/feature_evaluation/" + str(l) + "_scores.csv")

	print(stat_df.mean(1))
	print("\nGenerating taxa scores and generating network file...")
	network, tree_scores = generate_network(g, feature_scores, label_set)
	tree_scores.to_csv(result_path + "/tree_scores.csv")
	
	with open(result_path + '/network.json', 'w') as json_file:
		json.dump(network, json_file, sort_keys=True, indent=4, separators=(',', ': '))

	#####################################################################
	# Build Single Final Predictive Model
	#####################################################################
	final_x = MinMaxScaler().fit_transform(my_maps.reshape(-1, tree_row * tree_col)).reshape(-1,tree_row, tree_col)
	train = [final_x, labels_oh]

	train_weights = []

	for l in np.unique(labels):
		a = float(len(labels))
		b = 2.0 * float((np.sum(labels==l)))
		c_prob[int(l)] = a/b

	c_prob = np.array(c_prob).reshape(-1)

	for l in np.argmax(labels_oh, 1):
		train_weights.append(c_prob[int(l)])
	train_weights = np.array(train_weights)
	print("Building final predictive model...")
	popphy_model = PopPhyCNN((tree_row, tree_col), num_class, config)
	popphy_model.train(train, train_weights)					
	popphy_model.model.save(result_path + '/PopPhy-CNN.h5')


if __name__ == "__main__":
	train_PopPhy()
