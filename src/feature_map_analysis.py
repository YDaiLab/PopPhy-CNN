import sys
import pandas as pd
import numpy as np
import argparse
from popphy_io import load_network, load_data_from_file
from graph import Graph, Node

#Convert abundance vector into tree matrix
def generate_maps(x, g, f, p=-1):
        id = multiprocessing.Process()._identity
        g.populate_graph(f, x)
        return x, np.array(g.get_map(p)), np.array(g.graph_vector())


def get_feature_map_rankings(dset, x, y, pred, fm, w, b):

	prefix = "CV_"
	fm = np.array(fm)
	w = np.array(w)
	x = np.array(x)
	y = np.array(y)
        fm = np.squeeze(fm)
        w = np.squeeze(w)
        x = np.squeeze(x)

	#Get reference graph            
	g = Graph()
	g.build_graph("../data/" + dset + "/newick.txt")
	ref = g.get_ref()

	ref_val = np.zeros((ref.shape[0], ref.shape[1]))
	num_nodes = g.get_node_count()

	rankings = {}
	scores = {}
	node_names = g.get_dictionary()

	fp = open("../data/" + dset +"/label_reference.txt", 'r')
	labels = fp.readline().split("['")[1].split("']")[0].split("' '")
	fp.close()

	fp = open("../data/" + dset +"/otu.csv", 'r')
	otus = fp.readline().split(",")
	fp.close()

	num_classes = len(labels)
	
	for i in range(0, num_classes):
		rankings[i] = {}
	        scores[i] = {}
		for j in node_names:
		    rankings[i][j] = []
       		    scores[i][j] = []

	
	print(x.shape)
	total_num_samp = x.shape[0]
	num_maps = fm.shape[-1]
	w_row = w.shape[0]
	w_col = w.shape[1]

	data = {}
	fm_data = {}

	for i in range(0, num_classes):	
		data[i] = []
		fm_data[i] = []

	print(len(y))
	print(len(pred))
	for i in range(total_num_samp):
		print(".................")
		if y[i] == pred[i]:
			print(y[i])
			print(pred[i])
			l = y[i]
			data[l].append(x[i])
			fm_data[l].append(fm[i])


	num_samp = np.zeros(num_classes)

	for i in range(num_classes):
		num_samp[i] = int(len(data[i]))
		data[i] = np.array(data[i])
		fm_data[i] = np.array(fm_data[i])
	fm_rows = fm.shape[1]
	fm_cols = fm.shape[2]
	print(num_samp)
	theta1 = 1
	theta2 = 0

	#Get the top X max indices for each class and each feature map
	max_list = np.zeros((num_classes, num_maps, fm_rows * fm_cols))

	for i in range(num_classes):
		for j in range(0, int(num_samp[i])):
			for k in range(0, num_maps):
				maximums = np.argsort(fm_data[i][j,:,:,k].flatten())[::-1]
				for l in range(0, int(round(theta1 * num_nodes))):
					if (fm_data[i][j,:,:,k].flatten()[maximums[l]] > b[k]):
		        			max_list[i][k][maximums[l]] += 1
					else:
						break


	d = {"OTU":otus,"Max Score":np.zeros(len(otus)), "Cumulative Score":np.zeros(len(otus))}
	df = pd.DataFrame(data = d)
	results = {}


	for i in range(0, num_classes):
	    results[i] = df.set_index("OTU")

	for i in range(0, num_classes):
	   #For each feature map...
	   for j in range(0, num_maps):
		print(j)
		#Order the feature map's maximums
		loc_list = max_list[i][j].argsort()[::-1]    
		#Store kernel weights
		#For the top X maximums...i
		for k in range(0, len(loc_list)):
		    #Find the row and column location and isolate reference window
		    loc = loc_list[k]
		    max_count = max_list[i][j][loc]
		    if max_count == 0:
			break
		    if max_count >= int(round(num_samp[i] * theta2)):
		        row = loc / fm_cols
		        col = loc % fm_cols
		        ref_window = ref[row:row + w_row, col:col + w_col]                
		        count = np.zeros((w_row,w_col))
		        
		        #Calculate the proportion of the contribution of each pixel to the convolution with the absolute value of weights
		        for l in range(0,int(num_samp[i])):
		            	window =data[i][l, row:row + w_row, col:col + w_col]
		            	abs_v = (abs(w[:,:,j]) * window).sum() + 0.00001
				v = (w[:,:,j] * window)
		            	for m in range(0, v.shape[0]):
		                    for n in range(0, v.shape[1]):
		                    	count[m,n] += v[m,n] / abs_v
		        
		        #Divide by number of samples
		        count = count/num_samp[i]
	     
		        #Print out features with a high enough value    
		        for m in range(0, w_row):
		            for n in range(0, w_col):
		                if count[m,n] > 0:
		                        if ref_window[m,n] in results[i].index:
		                            if count[m,n] > results[i].loc[ref_window[m,n], "Max Score"]:
		                                results[i].loc[ref_window[m,n], "Max Score"] = count[m,n]
		                            results[i].loc[ref_window[m,n], "Cumulative Score"] += count[m,n]
		                        else:
		                            results[i].loc[ref_window[m,n], "Max Score"] = count[m,n]    
		                            results[i].loc[ref_window[m,n], "Cumulative Score"] = count[m,n]    

	diff = {}
	
	for i in range(0, num_classes):
	    diff[i] = df.set_index("OTU")
	    for j in results[i].index:
	       for k in range(0, num_classes):
		   if i != k:
		       if j in results[k].index:
		           diff[i].loc[j, "Max Score"] = results[i].loc[j, "Max Score"] - results[k].loc[j, "Max Score"]
		           diff[i].loc[j, "Cumulative Score"] = results[i].loc[j, "Cumulative Score"] - results[k].loc[j, "Cumulative Score"]
		       else:
		           diff[i].loc[j, "Max Score"] = results[i].loc[j, "Max Score"]     
		           diff[i].loc[j, "Cumulative Score"] = results[i].loc[j, "Cumulative Score"]     

            rank = diff[i]["Max Score"].rank(ascending=False)
 	    for j in node_names:
                if j in rank.index:
		    rankings[i][j] = rank.loc[j]
		    scores[i][j] = diff[i].loc[j,"Max Score"]
	        else:
                    rankings[i][j] = rank.shape[0] + 1
                    scores[i][j] = 0

	score_df = {}
	rank_df = {}
	for i in range(num_classes):
		score_df[i] = pd.DataFrame(scores[i].items(), columns=["OTU", "Score"])
		score_df[i] = score_df[i].set_index("OTU")
		rank_df[i] = pd.DataFrame(rankings[i].items(), columns=["OTU", "Rank"])
		rank_df[i] = rank_df[i].set_index("OTU")
	
	return score_df, rank_df
