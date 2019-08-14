import os
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_score, recall_score, f1_score
import configparser as cp
from scipy.stats import trim_mean
import numpy as np
import json
import re

def get_config():
	config = cp.ConfigParser()
	config.read('config.py')
	return config

def save_params(param_dict, path):
	with open(path + "/model_paramters.json", 'w') as f:
		json_out = json.dumps(param_dict, sort_keys=True, indent=4)
		json_out2 = re.sub(r'": \[\s+', '": [', json_out)
		json_out3 = re.sub(r',\s+', ', ', json_out2)
		json_out4 = re.sub(r'\s+\]', ']', json_out3)
		json_out5 = re.sub(r'\s\}, ', '},\n    ', json_out4)
		json_out6 = re.sub(r'\{\}, ', '{},\n    ', json_out5)
		json_out7 = re.sub(r'\], ', '],\n        ', json_out6)
		json_out8 = re.sub(', "', ',\n        "' , json_out7)
		f.write(json_out8)

def load_params(path):
	with open(path + "/model_paramters.json", 'r') as f:
		param_str = f.read()
	param_dict = json.loads(param_str)
	return(param_dict)

	
def get_stat(y, probs, metric):
	config = get_config()
	y = np.array(y)
	probs = np.array(probs)
	num_class = probs.shape[1]
	
	
	if num_class > 2:
		metric = 'MCC'

	if probs.ndim == 2:
		probs = np.expand_dims(probs, -1)

	if probs.shape[2] > 1:
		num_pred = probs.shape[2]
		trim_cut = 1.0 / float(num_pred)
		prob_mean = trim_mean(probs, trim_cut, axis=0)
		prob_round = (probs == probs.max(axis=-1)[:,None,:]).astype(int)
		prob_sum = np.sum(prob_round, axis=-1)
		prob_median = np.argmax(prob_sum, axis=1)
	else:
		prob_mean = probs
		prob_median = np.argmax(probs, axis=1)

	if metric == "AUC":
		lab = np.argmax(prob_mean, axis=1)
		stat = roc_auc_score(y, prob_mean[:,1], average='weighted')
		
	if metric == "MCC":
		lab = prob_median
		stat = matthews_corrcoef(y, prob_median)
		
	return stat
	
def get_stat_dict(y, probs):
	config = get_config()
	y = np.array(y)
	probs = np.array(probs)

	num_class = probs.shape[1]
	stat_dict = {}
	if num_class > 2:
		metric = 'MCC'

	if probs.ndim == 2:
		probs = np.expand_dims(probs, -1)

	if len(y.shape) > 1 and y.shape[-1] > 1:
		y = np.argmax(y, axis=1)
	if probs.shape[2] > 1:
		num_pred = probs.shape[2]
		trim_cut = 1.0 / float(num_pred)
		prob_mean = trim_mean(probs, trim_cut, axis=-1)
		prob_round = (probs == probs.max(axis=1)[:,None,:]).astype(int)
		prob_sum = np.sum(prob_round, axis=-1)
		prob_median = np.argmax(prob_sum, axis=1)
	else:
		prob_mean = probs
		prob_median = np.argmax(probs, axis=1)
			
	lab = np.argmax(prob_mean, axis=1)
	lab_vote = prob_median
	if num_class == 2:	
		stat_dict["AUC"] = roc_auc_score(y, prob_mean[:,1], average='weighted')
	stat_dict["MCC"] = matthews_corrcoef(y, prob_median.reshape(-1))
	stat_dict["Precision"] = precision_score(y, lab, average='weighted')
	stat_dict["Recall"] = recall_score(y, lab, average='weighted')
	stat_dict["F1"] = f1_score(y, lab, average='weighted')
	
	
	return stat_dict
