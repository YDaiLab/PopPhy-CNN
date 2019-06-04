from graph import Graph
import sys
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="PopPhy-CNN Training")
parser.add_argument("-d", "--data_set", default="Cirrhosis",     help="Name of dataset in data folder.")
args = parser.parse_args()

dset = args.data_set

g = Graph()
g.build_graph("../data/" + dset + "/newick.txt")
ref = g.get_ref()

g.write_table("../data/" + dset + "/tree.out")

fp = open("../data/" + dset +"/label_reference.txt", 'r')
labels = fp.readline().split("['")[1].split("']")[0].split("' '")
fp.close()

scores_df={}
quantile_rank_df={}

for l in labels:
    scores_df[l] = pd.read_csv("../data/" + dset + "/" + l + "_scores.csv", index_col=0)
    quantile_rank_df[l] = pd.read_csv("../data/" + dset + "/" + l +"_quantile_rankings.csv", index_col=0, header=None)


scores_diff = scores_df[labels[0]].subtract(scores_df[labels[1]])
sign_list = []
for i in scores_diff.index.values:
	if abs(quantile_rank_df[labels[0]].loc[i].values - quantile_rank_df[labels[1]].loc[i].values) <= 20:
		sign_list.append(0)
	elif quantile_rank_df[labels[0]].loc[i].values < quantile_rank_df[labels[1]].loc[i].values and quantile_rank_df[labels[0]].loc[i].values:
		sign_list.append(-1)
	else:
		sign_list.append(1)

tree_scores = scores_diff.abs().quantile(q=0.75, axis=1) * sign_list
joint_rank = scores_diff.abs().rank(axis=0, ascending=False).quantile(q=0.25, axis=1)


tree_scores.to_csv("../data/" + dset + "/tree_scores.csv")
joint_rank.to_csv("../data/" + dset + "/joint_rank.csv")

op = open("../data/" + dset + "/tree_edges.csv", 'w')      
fp = open("../data/" + dset + "/tree.out")

for line in fp:
    node1 = line.split("\t")[0]
    node2 = line.split("\t")[1].split("\n")[0]
    if np.abs(tree_scores.loc[node1]) > 0.05 and np.abs(tree_scores.loc[node2]) > 0.05 and tree_scores.loc[node1] * tree_scores.loc[node2] > 0:
        v = (tree_scores[node1] + tree_scores[node2])/2
    else:
        v = 0
    op.write(node1 + "," + node2 + "," + str(v) + "\n")
op.close()
fp.close()	
