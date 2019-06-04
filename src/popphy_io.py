import pickle
import os

def load_data_from_file(dset, method, set, split):
   
	if method == "CV": 
		dir = "../data/" + dset + "/data_sets/" + method + "_" + set + "/" + split
	elif method == "HO" or method == "holdout":
		dir = "../data/" + dset + "/data_sets/" + method + "_" + set
	else:
		dir = "../data/" + dset + "/data_sets/" + method + "_" + set + "/" + split
	f = open(dir+"/training.save", 'rb')
	train = pickle.load(f)
	f.close()
	
	f = open(dir+"/test.save", 'rb')
	test = pickle.load(f)
	f.close()
	
	f = open(dir+"/validation.save", 'rb')
	validation = pickle.load(f)
	f.close()
	
	return train, test, validation

def save_network(net, dir):
	f = open(dir + "/net.save", 'wb')
	pickle.dump(net, f)
	f.close()

def save_network_1D(net, dir):
	f = open(dir + "/net_1D.save", 'wb')
	pickle.dump(net, f)
	f.close()

def load_network(dir):
	f = open(dir+"/net.save", 'rb')
	dat = pickle.load(f)
	f.close()
	return dat

def load_network_1D(dir):
	f = open(dir + "/net_1D.save", "rb")
	dat = pickle.load(f)
	f.close()
	return dat	
