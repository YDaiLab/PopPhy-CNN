import numpy as np

DICT = {}

for line in open("abundance_t2d-WT2D.txt"):
    split = line.split("\t")
    otu = split[0].split("g__")
    print(otu)
    if len(otu) > 1:
	otu = otu[1]
    else:
        otu = otu[0].split("f__")[1]
    otu = otu.split("_noname")[0].split("_unclassified")[0].replace("_", " ").replace("XIII", "XIII.").strip()
    abun = np.array(split[1:], dtype=float)
    if otu in DICT:
        DICT[otu] = np.add(DICT[otu], abun)
    else:
        DICT[otu] = abun

otus = DICT.keys()
abuns = np.vstack(DICT.values()).T

print(len(otus))

data_file = open("count_matrix.csv", "w")
for i in range(0,len(abuns)):
    data_file.write(",".join(map(str, abuns[i])))
    data_file.write("\n")
data_file.close()

otu_file = open("otu.csv", "w")
otu_file.write(",".join(map(str,otus)))
otu_file.close()
