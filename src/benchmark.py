import numpy as np
import pandas as pd
import os
import sys
import struct
import argparse
from array import array as pyarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, ShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score,matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler, StandardScaler

parser = argparse.ArgumentParser(description="Prepare Data")
parser.add_argument("-m", "--method", default="CV", help="CV or Holdout method.") #Holdout method TBI
parser.add_argument("-d", "--dataset", default="Cirrhosis", 	help="Name of dataset in data folder.")
parser.add_argument("-n", "--splits", default=10, type=int, help="Number of cross validated splits.")
parser.add_argument("-s", "--sets", default=10, type=int, help="Number of datasets to generate")
parser.add_argument("-c", "--classifier", default="RF", help="Classifier to use.")
parser.add_argument("-t", "--stat", default="MCC", help="Statistic to evaluate models.")
parser.add_argument("-x", "--features", default="raw", help="Raw or Tree feature set")
parser.add_argument("-b", "--num_class", default=2, help="Number of classes")
parser.add_argument("-f", "--eval_features", default=False, help="Boolean to evaluate features")
parser.add_argument("-z", "--thresh", default=0.5, help="Threshold for feature evaluation")
args = parser.parse_args()

accuracy = []
roc_auc = []
mcc_list = []
precision = []
recall = []
f_score = []
pred_list = []
prob_list = []
method = args.method
total = args.splits * args.sets
data = args.dataset
classifier = args.classifier
stat = args.stat
feature_set = args.features
num_class = args.num_class
eval_features = args.eval_features
thresh = args.thresh

if feature_set == "raw":
	train_file = "/benchmark_train_data.csv"
	test_file = "/benchmark_test_data.csv"
if feature_set == "tree":
	train_file = "/benchmark_tree_train_data.csv"
	test_file = "/benchmark_tree_test_data.csv"

fi = {}
if feature_set == "raw":
	fp = open("../data/" + str(data) + "/original_features.csv", 'r')
if feature_set == "tree":
	fp = open("../data/" + str(data) + "/tree_features.csv", 'r')
features = fp.readline().split(",")
fp.close()

for f in features:
	fi[f]=0

for set in range(0,args.sets):
	for cv in range(0,args.splits):
		dir = "../data/" + str(data) + "/data_sets/CV_" + str(set) + "/" + str(cv)
		best_auc = 0
		best_mcc = -1
		best_acc = 0
		best_precision = 0
		best_recall = 0
		best_f1 = 0
		best_pred = []
		best_prob = []
		best_clf = None

		for run in range(0,1):
			run_auc = 0
			if classifier=="RF":
				x = np.loadtxt(dir + train_file, delimiter=',')
				y = np.loadtxt(dir+'/benchmark_train_labels.csv', delimiter=',')
				tx = np.loadtxt(dir + test_file,  delimiter=',')
				ty = np.loadtxt(dir+'/benchmark_test_labels.csv', delimiter=',')

				scaler = MinMaxScaler().fit(x)
				x = scaler.transform(x)
				tx = np.clip(scaler.transform(tx),0,1)
				clf = RandomForestClassifier(max_depth=None, n_estimators=500, min_samples_split=2, n_jobs=-1)

				clf.fit(x,y)
				prob = [row[1] for row in clf.predict_proba(tx)]
				pred = [row for row in clf.predict(tx)]

			if classifier=="SVM":
				x = pd.read_csv(dir + train_file, header=None, dtype=np.float64)
				y = pd.DataFrame(np.loadtxt(dir+'/benchmark_train_labels.csv', delimiter=','))
				tx = pd.read_csv(dir + test_file, header=None, dtype=np.float64)
				ty = pd.DataFrame(np.loadtxt(dir+'/benchmark_test_labels.csv', delimiter=','))
	
				x = x.values
				tx = tx.values

				scaler = MinMaxScaler().fit(x)
				x = np.clip(scaler.transform(x), 0, 1)
				tx = np.clip(scaler.transform(tx),0,1)

				cv_grid = [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']}, {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']} ]
				clf = GridSearchCV(SVC(C=1, probability=True), param_grid=cv_grid, cv=5, n_jobs=-1, scoring="roc_auc")
				clf.fit(x, y[0])
				prob = [row[1] for row in clf.predict_proba(tx)]
				pred = [row for row in clf.predict(tx)]
				

			if classifier=="LASSO":
				x = np.loadtxt(dir + train_file, delimiter=',')
				y = np.loadtxt(dir+'/benchmark_train_labels.csv', delimiter=',')
				tx = np.loadtxt(dir + test_file, delimiter=',')
				ty = np.loadtxt(dir+'/benchmark_test_labels.csv', delimiter=',')

				scaler = MinMaxScaler().fit(x)
				x = scaler.transform(x)
				tx = np.clip(scaler.transform(tx),0,1)
				
				clf = LassoCV(alphas=np.logspace(-4, -0.5, 50), cv=5, n_jobs=-1)
				clf.fit(x, y)
				prob = [row for row in clf.predict(tx)]
				pred = [int(i > 0.5) for i in prob]
		
			run_auc=0
			if num_class == 2:
				run_auc = roc_auc_score(ty, prob)
			run_mcc = matthews_corrcoef(ty, pred)

			best_auc = run_auc
			best_mcc = matthews_corrcoef(ty, pred)
			best_acc = clf.score(tx,ty)
			best_precision = precision_score(ty, pred, average='weighted')
			best_recall = recall_score(ty, pred, average='weighted')
			best_f = f1_score(ty, pred, average='weighted')
			best_pred = pred
			best_prob = prob
			best_clf = clf	


		accuracy.append(best_acc)
		roc_auc.append(best_auc)
		mcc_list.append(best_mcc)
		precision.append(best_precision)
		recall.append(best_recall)
		f_score.append(best_f)
		pred_list.append(best_pred)
		prob_list.append(best_prob)

		if eval_features:
			if classifier == "RF" and best_mcc > thresh:
				i=0
				for f in features:
					fi[f] += best_clf.feature_importances_[i]
					i += 1

			if classifier == "LASSO" and best_mcc > thresh:
				i=0
				for f in features:
					fi[f] += abs(best_clf.coef_[i])/sum(abs(best_clf.coef_))
					i += 1


if eval_features:		
	if classifier == "LASSO" or classifier == "RF":				
		for f in features:
			fi[f] = fi[f]/total

		fp = open("../data/" + str(data) + "/" + classifier + "_features.txt", 'w') 
		for key in sorted(fi):
			fp.write(str(key) + "\t" + str(fi[key]) + "\n")
		fp.close()

print("Accuracy = " + str(np.mean(accuracy)) + " (" + str(np.std(accuracy)) + ")\n")
print(accuracy)
print("\n\nROC AUC = " + str(np.mean(roc_auc)) + " (" + str(np.std(roc_auc)) + ")\n")
print(roc_auc)
print("\n\nMCC = " + str(np.mean(mcc_list)) + " (" + str(np.std(mcc_list)) + ")\n")
print(mcc_list)
print("\n\nPrecision = " + str(np.mean(precision)) + " (" + str(np.std(precision)) + ")\n")
print("Recall = " + str(np.mean(recall)) + " (" + str(np.std(recall)) + ")\n")
print("F1 = " + str(np.mean(f_score)) + " (" + str(np.std(f_score)) + ")\n")

dir = "../data/" + str(data)
f = open(dir + "/data_sets/" + method + "_results_" + classifier + "_" + feature_set +".txt", 'w')
f.write("Mean Accuracy: " + str(np.mean(accuracy)) + " (" + str(np.std(accuracy))+ ")\n")
f.write(str(accuracy) + "\n")
f.write("\nMean ROC: " + str(np.mean(roc_auc)) + " (" + str(np.std(roc_auc))+ ")\n")
f.write(str(roc_auc) + "\n")
f.write("\nMean MCC: " + str(np.mean(mcc_list)) + " (" + str(np.std(mcc_list))+ ")\n")
f.write(str(mcc_list) + "\n")
f.write("\nMean Precision: " + str(np.mean(precision)) + " (" + str(np.std(precision))+ ")\n")
f.write(str(precision) + "\n")
f.write("\nMean Recall: " + str(np.mean(recall)) + " (" + str(np.std(recall))+ ")\n")
f.write(str(recall) + "\n")
f.write("\nMean F-score: " + str(np.mean(f_score)) + " (" + str(np.std(f_score))+ ")\n")
f.write(str(f_score) + "\n")

for i in range(0,total):
	f.write("\nPredictions for " + str(i) + "\n")
	f.write("\n" + str(pred_list[i]) + "\n")
	f.write("\n" + str(prob_list[i]) + "\n")
f.close()
  
