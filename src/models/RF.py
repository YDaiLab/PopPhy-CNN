# Third-party libraries
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from utils.popphy_io import get_stat_dict, get_stat
from sklearn.metrics import roc_curve


class RF():
    def __init__(self, config):
        self.num_trees = int(config.get('Benchmark', 'NumberTrees'))
        self.model = RandomForestClassifier(n_estimators= self.num_trees, n_jobs=-1)
        self.num_valid_models = int(config.get('Benchmark', 'ValidationModels'))
        self.feature_importance = []
        self.features = []
        
    def train(self, train, seed=42):

        x, y = train
        self.model.fit(x, y)
        self.feature_importance = self.model.feature_importances_
        feature_ranking = np.flip(np.argsort(self.feature_importance))
        num_features = x.shape[1]
        best_num_features = num_features

        skf = StratifiedKFold(n_splits=self.num_valid_models, shuffle=True)
        best_score = -1

        if len(np.unique(y)) == 2:
            metric = "AUC"
        else:
            metric = "MCC"
        
        for percent in [0.25, 0.5, 0.75, 1.0]:
            run_score = -1
            run_probs = []
            for train_index, valid_index in skf.split(x, y):
                train_x, valid_x = x[train_index], x[valid_index]
                train_y, valid_y = y[train_index], y[valid_index]

                features_using = int(round(num_features * percent))
                feature_list = feature_ranking[0:features_using]
                filtered_train_x = train_x[:,feature_list]
                filtered_valid_x = valid_x[:,feature_list]
                clf = RandomForestClassifier(n_estimators= self.num_trees, n_jobs=-1).fit(filtered_train_x, train_y)
                probs = [row for row in clf.predict_proba(filtered_valid_x)]
                run_probs = list(run_probs) + list(probs)
            run_score = get_stat(y, run_probs, metric)

            if run_score > best_score:
                best_num_features = num_features

        self.feature_list = feature_ranking[0:best_num_features]
        x_filt = x[:,self.feature_list]

        self.model.fit(x_filt, y)
        return

    def test(self, test):
            x, y = test
            x_filt = x[:,self.feature_list]
            probs = np.array([row for row in self.model.predict_proba(x_filt)])
            preds = np.argmax(probs, axis=-1)
            stat = get_stat_dict(y, probs)
            return preds, stat
