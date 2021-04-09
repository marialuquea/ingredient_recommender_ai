from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

class RandomForest:
    def __init__(self, xTrain, yTrain, xTest, yTest, verbose=True):
        print("\n\nRunning Random Forest classifier...")

        rf = RandomForestClassifier(random_state=42, verbose=verbose).fit(xTrain.values, yTrain.values)
        self.clf = rf
        self.feature_list = list(xTrain.columns)
        self.predictions = rf.predict(xTest)
        self.accuracy = accuracy_score(yTest.values, self.predictions)

    def predict(self, x):
        return self.clf.predict(x)

    def get_importances(self):
        importances = list(self.clf.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(self.feature_list, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    def grid_search(self, X_train, y_train):
        param_grid = {'n_estimators': [500, 1000, 2000],
                      'min_samples_split': [2, 10, 20],
                      'min_samples_leaf': [1, 10, 100],
                      'max_features': ['auto', 5, 'sqrt', 'log2', None]
                      }
        grid = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=3).fit(X_train, y_train)
        print("\nBest params:", grid.best_params_)
        print("\nBest score:", grid.best_score_)
        return grid.best_params_
