from itertools import cycle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from scipy import interp

class RandomForest:
    def __init__(self, xTrain, yTrain, xTest, yTest):
        print("Running Random Forest classifier...")

        rf = RandomForestClassifier(random_state=42).fit(xTrain.values, yTrain.values)
        self.rf = rf
        self.feature_list = list(xTrain.columns)
        self.predictions = rf.predict(xTest)
        self.errors = abs(self.predictions - yTest.values)
        self.accuracy = accuracy_score(yTest.values, self.predictions)

    def predict(self, x):
        return self.rf.predict(x)

    def get_importances(self):
        importances = list(self.rf.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(self.feature_list, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    def plotRandomForest(self, y_true, predictions):
        plt.scatter(y_true, predictions, c='#FF7AA6')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Precision of predicted outcomes')
        plt.plot(np.unique(y_true), np.poly1d(np.polyfit(y_true, predictions, 1))(np.unique(y_true)))
        plt.show()

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

    def evaluate_RF_model(self, xTrain, yTrain):
        kfold = KFold(n_splits=3, shuffle=True)
        count, avg_roc_auc, avg_accuracy, avg_precision, avg_recall, avg_f1score = 0, 0, 0, 0, 0, 0

        for train, test in kfold.split(xTrain):
            print(f"Test: {count + 1}")
            X_train, X_test = xTrain.iloc[train], xTrain.iloc[test]
            y_train, y_true = yTrain.iloc[train], yTrain.iloc[test]

            y_pred = self.rf.predict(X_test)
            avg_accuracy += accuracy_score(y_true, y_pred)
            avg_precision += precision_score(y_true, y_pred, average='micro')
            avg_recall += recall_score(y_true, y_pred, average='micro')
            avg_f1score += f1_score(y_true, y_pred, average='micro')
            count += 1
        avg_accuracy /= count
        avg_precision /= count
        avg_recall /= count
        avg_f1score /= count
        print("\nRF evaluation results")
        print("Average accuracy:", avg_accuracy)
        print("Average precision:", avg_precision)
        print("Average recall:", avg_recall)
        print("Average f1 score:", avg_f1score)

        return avg_accuracy, avg_precision, avg_recall, avg_f1score

    def plot_roc_curve(self, X_train, y_train, X_test, y_test):
        y = label_binarize(y_train, classes=np.unique(y_train))
        y_test = label_binarize(y_test, classes=np.unique(y_train))
        n_classes = y.shape[1]
        rf = OneVsRestClassifier(RandomForestClassifier()).fit(X_train, y)
        y_score = rf.predict(X_test)
        print('y_score shape',y_score.shape)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"],label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic in multi-class RF model')
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        plt.show()



