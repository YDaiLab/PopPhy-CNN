import numpy as np
import os
import struct
from array import array as pyarray
from numpy import unique
from graph import Graph
from random import shuffle
from random import seed
from joblib import Parallel, delayed
import multiprocessing
import time
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import pandas as pd
import argparse
import pickle
from copy import deepcopy

parser = argparse.ArgumentParser(description="Prepare Data")
parser.add_argument("-m", "--method", default="CV", help="CV or Holdout method.") #Holdout method TBI
parser.add_argument("-d", "--dataset", default="Cirrhosis", help="Name of dataset in data folder.")
parser.add_argument("-n", "--splits", default=10, type=int, help="Number of cross validated splits.")
parser.add_argument("-s", "--sets", default=10, type=int, help="Number of datasets to generate")
args = parser.parse_args()


#Convert abundance vector into tree matrix
def generate_maps(x, g, f, v, p=-1):
	id = multiprocessing.Process()._identity
	temp_g = deepcopy(g)
	temp_g.populate_graph(f, x)
	map = temp_g.get_map()
	vector = temp_g.graph_vector()
	raw_vector = temp_g.raw_vector(v)
	del(temp_g)
	return np.array(raw_vector), np.array(map), np.array(vector)

def get_feature_df(features):
	kingdom, phylum, cl, order, family, genus, species  = [], [], [], [], [], [], []
	for f in features:

		name = f.split("k__")[1].split("|p__")[0].replace(".","")
		if "_unclassified" in name:
			name = 'unclassified_' + name.split("_unclassified")[0]
		kingdom.append(name)

		if "p__" in f:
			name =f.split("p__")[1].split("|c__")[0].replace(".","")
			if "_unclassified" in name:
				name = 'unclassified_' + name.split("_unclassified")[0]
			phylum.append(name)
		else:
			phylum.append("NA")

		if "c__" in f:
			name = f.split("c__")[1].split("|o__")[0].replace(".","")
			if "_unclassified" in name:
				name = 'unclassified_' + name.split("_unclassified")[0]
			cl.append(name)
		else:
			cl.append("NA")

		if "o__" in f:
			name = f.split("o__")[1].split("|f__")[0].replace(".","")
			if "_unclassified" in name:
				name = 'unclassified_' + name.split("_unclassified")[0]
			order.append(name)
		else:
			order.append("NA")

		if "f__" in f:
			name = f.split("f__")[1].split("|g__")[0].replace(".","")
			if "_unclassified" in name:
				name = 'unclassified_' + name.split("_unclassified")[0]
			family.append(name)
		else:
			family.append("NA")

		if "g__" in f:
			name = f.split("g__")[1].split("|s__")[0].replace(".","")
			if "_unclassified" in name:
				name = 'unclassified_' + name.split("_unclassified")[0]
			genus.append(name)
		else:
			genus.append("NA")

		if "s__" in f:
			name = f.split("s__")[1]
			if "_unclassified" in name:
				name = 'unclassified_' + name.split("_unclassified")[0]
			species.append(name)
		else:
			species.append("NA")

	if len(species) == 0:
		d = {'kingdom': kingdom, 'phylum': phylum, 'class':cl,
			'order':order, 'family':family, 'genus':genus}
		feature_df = pd.DataFrame(data=d)
		feature_df.index = feature_df['genus']
	else:
		d = {'kingdom': kingdom, 'phylum': phylum, 'class':cl,
			'order':order, 'family':family, 'genus':genus, 'species': species}
		feature_df = pd.DataFrame(data=d)
		feature_df.index = feature_df['species']
	return feature_df

data = args.dataset
num_cores = multiprocessing.cpu_count()

if __name__ == "__main__":
	holdout_seed = 0
	dat_dir = "../data/" + data
	holdout_seed = holdout_seed + 1
	print("Processing " + data +"...")
		
	#Data Variables	
	my_x = []
	my_y = []
		
				
	data = pd.read_csv(dat_dir + '/abundance.tsv', index_col=0, sep='\t', header=None)
	labels = np.genfromtxt(dat_dir + '/labels.txt', dtype=np.str_, delimiter=',')

	data = data.transpose()

	sums = data.sum(axis=1)
	data = data.divide(sums, axis=0)
	labels, label_set = pd.factorize(labels)	
	print("Finished reading data...")
		
	#Get the set of classes	
	num_classes = len(label_set)
	print("There are " + str(num_classes) + " classes")  
	f = open(dat_dir + "/label_reference.txt", 'w')
	f.write(str(label_set))
	f.close()

	features = list(data.columns.values)
	features_df = get_feature_df(features)
	#Build phylogenetic tree graph
	g = Graph()
	g.build_graph(dat_dir + "/newick.txt")
	ref = g.get_ref()
	print("Graph constructed...")
	#Create 10 CV sets
	tree_features = []
	raw_features = []
	for m in range(ref.shape[0]):
		for n in range(ref.shape[1]):
			r = ref[m,n]
			if r != 0:
				tree_features.append(r)
			if r in features_df['genus'].values:
				raw_features.append(r)
			if r in features_df['family'].values and ("" in features_df.loc[features_df["family"]==r]['genus'].values or "NA" in features_df.loc[features_df["family"]==r]['genus'].values):
				raw_features.append(r)
			if r in features_df['order'].values and ("" in features_df.loc[features_df["order"]==r]['family'].values or "NA" in features_df.loc[features_df["order"]==r]['family'].values):
				raw_features.append(r)
			if r in features_df['class'].values and ("" in features_df.loc[features_df["class"]==r]['order'].values or "NA" in features_df.loc[features_df["class"]==r]['order'].values):
				raw_features.append(r)
			if r in features_df['phylum'].values and ("" in features_df.loc[features_df["phylum"]==r]['class'].values or "NA" in features_df.loc[features_df["phylum"]==r]['class'].values):
				raw_features.append(r)
			if r in features_df['kingdom'].values and ("" in features_df.loc[features_df["kingdom"]==r]['phylum'].values or "NA" in features_df.loc[features_df["kingdom"]==r]['phylum'].values):
				raw_features.append(r)
	num_features = len(raw_features)	
	np.savetxt(dat_dir + "/tree_features.csv", tree_features, fmt='%s', delimiter=",")
	np.savetxt(dat_dir + "/original_features.csv", raw_features, fmt='%s', delimiter=",")
	
	features_df.to_csv(dat_dir + "/taxonomy.tsv", sep="\t", header=True)
	my_maps = []
	my_benchmark = []
	results = Parallel(n_jobs=num_cores)(delayed(generate_maps)(x,g,features_df, raw_features) for x in data.values)
	my_maps = np.array(np.take(results,1,1).tolist())
	my_benchmark = np.array(np.take(results,0,1).tolist())
	my_benchmark_tree = np.array(np.take(results,2,1).tolist())
	my_maps = np.array(my_maps)
	my_benchmark = np.array(my_benchmark)				
	my_benchmark_tree = np.array(my_benchmark_tree)				
	map_rows = my_maps.shape[1]
	map_cols = my_maps.shape[2]
	print("Finished Setup...") 
		
	for set in range(0,args.sets):
		print("Generating set " + str(set) + "...")
		if args.method=="CV":
			k_fold = StratifiedKFold(n_splits=args.splits, shuffle=True)		
			count = 0
			#for train_index, test_index in kf.split(my_maps[0], my_lab):
			for train_index, test_index in k_fold.split(data.values, labels):
				x_train = []
				x_test = []
				y_train=[]
				y_test=[]
				benchmark_train=[]
				benchmark_test=[]
				benchmark_tree_train=[]
				benchmark_tree_test=[]
				benchmark_tree_map_train=[]
				benchmark_tree_map_test=[]
		
				print("Creating split " + str(count))
			
				x_train.append(my_maps[train_index])
				x_test.append(my_maps[test_index])
				y_train.append(labels[train_index])
				y_test.append(labels[test_index])
				benchmark_train.append(my_benchmark[train_index])
				benchmark_test.append(my_benchmark[test_index])
				benchmark_tree_train.append(my_benchmark_tree[train_index])
				benchmark_tree_test.append(my_benchmark_tree[test_index])				

	
				x_train = np.array(x_train).reshape(len(train_index), map_rows, map_cols)
				x_test = np.array(x_test).reshape(len(test_index), map_rows, map_cols)
				y_train = np.squeeze(np.array(y_train).reshape(1,len(train_index)), 0)
				y_test = np.squeeze(np.array(y_test).reshape(1,len(test_index)), 0)
				benchmark_train = np.array(benchmark_train).reshape(len(train_index), num_features)
				benchmark_test = np.array(benchmark_test).reshape(len(test_index), num_features)
				benchmark_tree_train = np.array(benchmark_tree_train).reshape(len(train_index), -1)
				benchmark_tree_test = np.array(benchmark_tree_test).reshape(len(test_index), -1)
				benchmark_tree_map_train = []
				benchmark_tree_map_test = []

	

				benchmark_tree_map_train = np.array(benchmark_tree_map_train).reshape(len(train_index), -1)
				benchmark_tree_map_test = np.array(benchmark_tree_map_test).reshape(len(test_index), -1)

				seed = np.random.randint(1000)
				np.random.seed(seed)
				np.random.shuffle(x_train)
				np.random.seed(2*seed)
				np.random.shuffle(x_test)
				np.random.seed(seed)
				np.random.shuffle(y_train)
				np.random.seed(2*seed)
				np.random.shuffle(y_test)
				np.random.seed(seed)
				np.random.shuffle(benchmark_train)
				np.random.seed(2*seed)
				np.random.shuffle(benchmark_test)
				np.random.seed(seed)
				np.random.shuffle(benchmark_tree_train)
				np.random.seed(2*seed)
				np.random.shuffle(benchmark_tree_test)
				np.random.seed(seed)
				np.random.shuffle(benchmark_tree_map_train)
				np.random.seed(2*seed)
				np.random.shuffle(benchmark_tree_map_test)
			
				#Combine data and labels into a single object
				
				dir = dat_dir + "/data_sets//CV_" + str(set) + "/" + str(count)
				if not os.path.exists(dir):
					os.makedirs(dir)
				
				#Save the data sets in Pickle format
				f = open(dir + "/training.save", 'wb')
				pickle.dump(x_train, f)
				f.close()
				
				f = open(dir + "/test.save", 'wb')
				pickle.dump(x_test, f)
				f.close()
				
				f = open(dir + "/validation.save", 'wb')
				pickle.dump(x_test, f)
				f.close()
				
				
				np.savetxt(dir + "/benchmark_train_data.csv", benchmark_train, delimiter=',')
				np.savetxt(dir + "/benchmark_tree_train_data.csv", benchmark_tree_train, delimiter=',')
				np.savetxt(dir + "/benchmark_train_labels.csv", y_train, delimiter=',')
				np.savetxt(dir + "/benchmark_test_data.csv", benchmark_test, delimiter=',')
				np.savetxt(dir + "/benchmark_tree_test_data.csv", benchmark_tree_test, delimiter=',')
				np.savetxt(dir + "/benchmark_test_labels.csv", y_test, delimiter=',')
				count = count + 1


		print("Finished Setup...")

		if args.method=="holdout" or args.method=="HO":						
			k_fold = StratifiedShuffleSplit(labels, n_iter=1,  test_size=0.3, random_state=holdout_seed)
			count = 0
			#for train_index, test_index in kf.split(my_maps[0], my_lab):
			for train_index, test_index in k_fold:
				x_train = []
				x_test = []
				x_valid = []
				y_train=[]
				y_test=[]
				y_valid = []

				benchmark_train=[]
				benchmark_test=[]
				benchmark_valid=[]


				test_valid_split = int(round(len(test_index) * 0.667))
				valid_index = test_index[(test_valid_split + 1):]
				test_index = test_index[:test_valid_split]

				print("Creating split " + str(count))

				x_train.append(my_maps[train_index])
				x_test.append(my_maps[test_index])
				x_valid.append(my_maps[valid_index])
				y_train.append(labels[train_index])
				y_test.append(labels[test_index])
				y_valid.append(labels[valid_index])
				benchmark_train.append(my_benchmark[train_index])
				benchmark_test.append(my_benchmark[test_index])
				benchmark_valid.append(my_benchmark[valid_index])

				x_train = np.array(x_train).reshape(-1, map_rows, map_cols)
				x_test = np.array(x_test).reshape(-1, map_rows, map_cols)
				x_valid = np.array(x_valid).reshape(-1, map_rows, map_cols)
				y_train = np.squeeze(np.array(y_train).reshape(1,-1), 0)
				y_test = np.squeeze(np.array(y_test).reshape(1,-1), 0)
				y_valid = np.squeeze(np.array(y_valid).reshape(1,-1), 0)
				benchmark_train = np.array(benchmark_train).reshape(-1, num_features)
				benchmark_test = np.array(benchmark_test).reshape(-1, num_features)
				benchmark_valid = np.array(benchmark_valid).reshape(-1, num_features)
 
				
				dir = dat_dir + "/data_sets/HO_" + str(set) + "/"
				if not os.path.exists(dir):
					os.makedirs(dir)

				#Save the data sets in Pickle format
				f = open(dir + "/training.save", 'wb')
				pickle.dump(x_train, f)
				f.close()

				f = open(dir + "/test.save", 'wb')
				pickle.dump(x_test, f)
				f.close()

				f = open(dir + "/validation.save", 'wb')
				pickle.dump(x_valid, f)
				f.close()

				np.savetxt(dir + "/benchmark_train_data.csv", benchmark_train, delimiter=',')
				np.savetxt(dir + "/benchmark_train_labels.csv", y_train, delimiter=',')
				np.savetxt(dir + "/benchmark_test_data.csv", benchmark_test, delimiter=',')
				np.savetxt(dir + "/benchmark_test_labels.csv", y_test, delimiter=',')
				np.savetxt(dir + "/benchmark_valid_data.csv", benchmark_valid, delimiter=',')
				np.savetxt(dir + "/benchmark_valid_labels.csv", y_valid, delimiter=',')
				count = count + 1
	
			print("Finished writing files...")
