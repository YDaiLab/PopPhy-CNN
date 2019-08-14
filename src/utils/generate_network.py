from utils.graph import Graph, Node
import sys
import numpy as np
import pandas as pd
import json

def generate_network(g, scores, labels):
	if len(labels) > 2:
		print("Visualization of non-binary datasets not supported")

	ref = g.get_ref()


	scores_diff = scores[labels[0]].subtract(scores[labels[1]])
	quantiles = {}
	for l in labels:
		quantiles[l] = scores[l].rank(axis=0).quantile(q=0.25, axis=1)

	sign_list = []
	for i in scores_diff.index.values:
		if abs(quantiles[labels[0]].loc[i] - quantiles[labels[1]].loc[i]) <= 20:
			sign_list.append(0)
		elif quantiles[labels[0]].loc[i] < quantiles[labels[1]].loc[i] and quantiles[labels[0]].loc[i]:
			sign_list.append(-1)
		else:
			sign_list.append(1)

	tree_scores = scores_diff.abs().quantile(q=0.75, axis=1) * sign_list
   
	layers, width = g.get_size()
    
	nodes = []
	edges = []
	node_check = []
	edge_check = []
	for l in range(1,layers+1):
		for node in g.get_nodes(l):
			node_id = node.get_id()
			score = tree_scores.loc[node_id]
			node_element = {"data" : {"id":node_id, "score":score}}
			nodes.append(node_element)
			for c in node.get_children_ids():
				child_score = tree_scores.loc[c]
				if np.abs(score) > 0.05 and np.abs(child_score) > 0.05 and score * child_score > 0:
					edge_score = (score + child_score)/2
				else:
					edge_score = 0
				edge_element = {"data":{"id":node_id+"_"+c,"source":node_id, "target":c, "score":edge_score}}
				edges.append(edge_element)
	network = {"elements":{"nodes":list(np.array(nodes).reshape(-1)), "edges":list(np.array(edges).reshape(-1))}}
	for t in edge_check:
		if t not in node_check:
			print("E",t)
            
	for t in node_check:
		if t not in edge_check:
			print("N",t)
       
	return network, tree_scores

