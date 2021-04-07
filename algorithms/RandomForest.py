from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np

class RandomForest:
    def __init__(self, xTrain, yTrain, xTest, yTest):
        print("Running Random Forest classifier...")

        rf = RandomForestClassifier(random_state=42).fit(xTrain.values, yTrain.values)
        self.rf = rf
        self.feature_list = list(xTrain.columns)
        self.predictions = rf.predict(xTest)
        self.accuracy = accuracy_score(yTest.values, self.predictions)

    def predict(self, x):
        return self.rf.predict(x)

    def get_importances(self):
        importances = list(self.rf.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(self.feature_list, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    def gridSearch(self, X_train, y_train):
        param_grid = {'n_estimators': [500, 1000, 2000],
                      'criterion': ['mse', 'mae'],
                      'min_samples_split': [2, 10, 20],
                      'min_samples_leaf': [1, 10, 100],
                      'max_features': ['auto', 5, 'sqrt', 'log2', None],
                      'bootstrap': [True, False],
                      'oob_score': [True, False],
                      'warm_start': [True, False]
                      }
        grid = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=3).fit(X_train, y_train)
        print("\nBest params:", grid.best_params_)
        print("\nBest score:", grid.best_score_)
        return grid.best_params_
