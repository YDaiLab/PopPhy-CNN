# Third-party libraries
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC	
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize
from sklearn.metrics import roc_curve
from utils.popphy_io import get_stat_dict


class SVM():
    def __init__(self, config, label_set):
        self.num_valid_models = int(config.get('Benchmark', 'ValidationModels'))
        self.max_iter = int(config.get('Benchmark', 'MaxIterations'))
        self.num_class = len(label_set)
        self.classes = label_set
        if self.num_class > 2:
            self.grid = [{'estimator__C': [1, 10, 100, 1000], 'estimator__kernel': ['linear']},
                         {'estimator__C': [1, 10, 100, 1000], 'estimator__gamma': [0.001, 0.0001], 
                          'estimator__kernel': ['rbf']}]
            self.model = GridSearchCV(OneVsRestClassifier(SVC(probability=True, max_iter=self.max_iter)), 
                                      self.grid, cv= self.num_valid_models, n_jobs=-1, error_score='raise')
        else:
            self.grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                         {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
            self.model = GridSearchCV(SVC(probability=True, max_iter=self.max_iter), self.grid, 
                                  cv= self.num_valid_models, n_jobs=-1, scoring="roc_auc", error_score='raise')
    def train(self, train):
        x, y = train

        if self.num_class > 2:
            y = label_binarize(y, classes=self.classes)

        self.model.fit(x, y)
        
    def test(self, test):
        x, y = test
        if self.num_class > 2:
            y = label_binarize(y, classes=self.classes)

        probs = np.array([row for row in self.model.predict_proba(x)])
        preds = np.argmax(probs, axis=-1)
        stat= get_stat_dict(y, probs)
        return preds, stat    
