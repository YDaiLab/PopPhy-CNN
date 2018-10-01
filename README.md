# PopPhy-CNN

PopPhy-CNN,a novel convolutional neural networks (CNN) learning architecture that effectively exploits phylogentic structure in microbial taxa. PopPhy-CNN provides an input format of 2D matrix created by embedding the phylogenetic tree that is populated with the relative abundance of microbial taxa in a metagenomic sample. This conversion empowers CNNs to explore the spatial relationship of the taxonomic annotations on the tree and their quantitative characteristics in metagenomic data.


## Publication:
* Derek Reiman, Ahmed A. Metwally, Yang Dai. "PopPhy-CNN: A Phylogenetic Tree Embedded Architecture for Convolution Neural Networks for Metagenomic Data", bioRxiv, 2018.  [[paper](https://www.biorxiv.org/content/early/2018/01/31/257931)]

## Execution:

### Prerequisites
  - Python 2.7.14
  - Libraries: `pip install theano numpy pandas joblib xmltodict untangle sklearn network`
  
### To generate 10 times 10-fold cross validation sets for the Cirrhosis dataset:

```bash
python prepare_data.py -d=Cirrhosis -m=CV -n=10 -s=10
``` 

### To train PopPhy-CNN using the generated 10 times 10-fold cross validation Cirrhosis sets for 400 epochs with early stopping of 20 epochs:
```bash
python train.py -d=Cirrhosis -m=CV -n=10 -s=10 -e=400 -p=20
```

### To extract feature importance scores from the learned models:
```bash
python feature_map_analysis -d=Cirrhosis -m=CV -n=10 =s=10
```

### To generate files to use for Cytoscape visualization:
```bash
python generate_tree_scores.py -d=Cirrhosis
```


