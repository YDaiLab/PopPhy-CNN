# Third-party libraries
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize
from utils.popphy_io import get_stat_dict
from sklearn.metrics import roc_curve

class LASSO():
    
    def __init__(self, config, label_set):
        self.max_iter = int(config.get('Benchmark', 'MaxIterations'))
        self.num_cv = int(config.get('Benchmark', 'ValidationModels'))
        self.num_class = len(label_set)
        self.classes = label_set
        if self.num_class > 2:
            self.model = OneVsRestClassifier(LassoCV(alphas=np.logspace(-4, -0.5, 50), 
                                                     cv=self.num_cv, n_jobs=-1, max_iter=self.max_iter))
        else:
            self.model = LassoCV(alphas=np.logspace(-4, -0.5, 50), cv=self.num_cv, n_jobs=-1, max_iter=self.max_iter)
        
    def train(self, train):
        x, y = train
        if self.num_class > 2:
            y = label_binarize(y, classes=self.classes)
        self.model.fit(x,y)
        return
    
    def test(self, test):
        x, y = test
        if self.num_class > 2:
            y = label_binarize(y, classes=self.classes)
            
        if self.num_class == 2:
            probs = np.array([[1-row, row] for row in self.model.predict(x)])
            preds = np.argmax(probs, axis=-1)
            stats = get_stat_dict(y, probs)
        else:
            probs = self.model.predict(x)
            preds = np.argmax(probs, axis=-1)
            stats = get_stat_dict(y, probs)
           
        return preds, stats