import pandas as pd
import numpy as np

df = pd.read_csv("abundance.csv", index_col=0, sep="\t")
otus = list(df.columns.values)

new_names = otus
temp = []

for prefix in ["g__", "f__", "o__", "c__", "p__", "k__"]:
	print(prefix)
	for i in range(0,len(otus)):
		line = new_names[i].split(prefix)
		print(line)
		if line[-1] == "":
			print("appending " + line[-2])
			temp.append(line[-2])
		else:
			print("appending " + line[-1])
			temp.append(line[-1])
	new_names = temp
	temp = []

for i in range(0, len(otus)):
	new_names[i] = new_names[i].replace(".Other", "")
	new_names[i] = new_names[i].replace(".", "")

df.columns = new_names



df = df.drop([".", "" ,"BS11","57N15","S247","Paraprevotellaceae","Weeksellaceae","Gemellales","Gemellaceae","Gemellaceae","02d06","SMB53","Dehalobacteriaceae","EtOH8","Pseudoramibacter_Eubacterium","rc44","Mogibacteriaceae","Mogibacteriaceae","WAL_1855D","ph2","cc_115","03196G20","Cerasicoccaceae", "CF231", "mitochondria", "Ellin6075"], axis=1)
df.groupby(df.columns, axis=1).sum()

otus = list(df.columns.values)
print(otus)

df.to_csv("count_matrix.csv", header=False, index=False)

