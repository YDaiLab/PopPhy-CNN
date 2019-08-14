#####################################################################################
# Global settings
#####################################################################################

[Evaluation]

# NumberTestSplits	Number of partitions (k) in k-fold cross-validation [integer]
# NumberRuns		Number of iterations to run cross-validation [integer]
# Dataset		Dataset located in ../data
# FilterThresh		Filter OTUs based on porportion found in samples [float]

NumberTestSplits = 10
NumberRuns = 1
DataSet = Cirrhosis
FilterThresh = 0.1

#####################################################################################
# Benchmark settings
#####################################################################################

[Benchmark]

# Normalization		Normalization method [MinMax or Standard]

Normalization = MinMax

# NumberTrees		Number of trees per RF [integer]
# ValidationModels	Number of partitions (k) to use in internal cross-validation for tuning[integer]
# NumberIterations	Max iterations for LASSO and SVM training [integer]

NumberTrees = 500
ValidationModels = 5
MaxIterations = 10000

NumberKernel_1DCNN = 32
KernelWidth_1DCNN = 10
NumFCNodes_1DCNN = 32
NumConvLayers_1DCNN = 2
NumFCLayers_1DCNN = 2
L2_Lambda_1DCNN = 0.001
Dropout_1DCNN = 0.3
Patience_1DCNN = 40
LearningRate_1DCNN = 0.001
Batchsize_1DCNN = 1024

NumFCNodes_MLPNN = 32
NumFCLayers_MLPNN = 2
L2_Lambda_MLPNN = 0.001
Dropout_MLPNN = 0.3
Patience_MLPNN = 40
LearningRate_MLPNN = 0.001
Batchsize_MLPNN = 1024

#####################################################################################
# PopPhy-CNN models
#####################################################################################

[PopPhy]

# LearningRate		Learning rate for PopPhy-CNN models [float]
# BatchSize		Batch size for PopPhy-CNN models [integer]

LearningRate = 0.001
BatchSize = 1024
Patience = 40
NumberKernel = 32
KernelWidth = 10
KernelHeight = 3
NumFCNodes = 32
NumConvLayers = 1
NumFCLayers = 1
L2Lambda = 0.001
Dropout = 0.3
