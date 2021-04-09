from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

class SVM:
    def __init__(self, xTrain, yTrain, xTest, yTest, verbose=True):
        print("\n\nRunning Support Vector Machine classifier...")

        svm = SVC(kernel='linear', verbose=verbose).fit(xTrain, yTrain)
        self.clf = svm
        self.predictions = svm.predict(xTest)
        self.accuracy = accuracy_score(yTest.values, self.predictions)

    def predict(self, x):
        return self.clf.predict(x)

    def grid_search(selfself, X_train, y_train):
        param_grid = {'C': [0.1, 1, 2, 5],
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'shrinking': [True, False],
                      'tol': [1e-3, 0.01, 0.1]
                      }
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3).fit(X_train, y_train)
        print("\nBest params:", grid.best_params_)
        print("\nBest score:", grid.best_score_)
        return grid.best_params_

