# PopPhy-CNN

PopPhy-CNN,a novel convolutional neural networks (CNN) learning architecture that effectively exploits phylogentic structure in microbial taxa. PopPhy-CNN provides an input format of 2D matrix created by embedding the phylogenetic tree that is populated with the relative abundance of microbial taxa in a metagenomic sample. This conversion empowers CNNs to explore the spatial relationship of the taxonomic annotations on the tree and their quantitative characteristics in metagenomic data.


## Publication:
* Reiman D, Metwally AA, Sun J, Dai Y. PopPhy-CNN: A Phylogenetic Tree Embedded Architecture for Convolutional Neural Networks to Predict Host Phenotype From Metagenomic Data. IEEE J Biomed Health Inform. 2020 Oct;24(10):2993-3001. doi: 10.1109/JBHI.2020.2993761. Epub 2020 May 11. PMID: 32396115. [[paper](https://pubmed.ncbi.nlm.nih.gov/32396115/)]

## Execution:

We provide a python environment which can be imported using the [Conda](https://www.anaconda.com/distribution/) python package manager.

Deep learning models are built using [Tensorflow](https://www.tensorflow.org/). PopPhy-CNN has been updated to use **Tensorflow v1.14.0**.

To fully utilize GPUs for faster training of the deep learning models, users will need to be sure that both [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) and [cuDNN](https://developer.nvidia.com/cudnn) are properly installed.

Other dependencies should be downloaded upon importing the provided environment.

### Clone Repository
```bash
git clone https://github.com/YDaiLab/PopPhy-CNN.git
cd PopPhy-CNN
```

### Import Conda Environment

```bash
conda env create -f PopPhy.yml
source activate PopPhy
cd src
``` 
  
### Set Configuration Parameters:

Edit config.py to customize your PopPhy-CNN execution. Datasets need to be placed in their own folder within the data/ directory. There needs to be an abundance file in which each column is a sample and each row is a taxon structured following the example below:

```bash
k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Actinomycetales|f__Actinomycetaceae|g__Actinomyces|s__Actinomyces_graevenitzii
```

In this example, the taxa is *Actinomyces graevenitzii* and comes from the Bacteria kingdom, Actinobacteria phylum, Actinobacteria class, Actinomycetales order, Actinoycetaceae family, *Actinomyces* genus, and *graevenitzii* species. Note that the 's__' identifier should include the genus and species.

### Run PopPhy-CNN:

Once the configuration file is set, PopPhy-CNN is executed with

```bash
python train.py
```

Results are saved in the results directory under a subdirectory with the same name as the dataset's folder.

### Visualizing the Results

Cytoscape can be used to visualize the results from PopPhy-CNN's analysis. To do so, install and run [Cytoscape](https://cytoscape.org/). In the results timestamped folder, load the file 'network.json' into cytoscape. Then import the Cytoscape style found 'style.xml' found in the 'cytoscape_style' directory. It may also be useful to install the yFiles layouts and visualize the tree using the yFile radial layout.


