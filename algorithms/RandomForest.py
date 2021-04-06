from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

    def plotRandomForest(self, xTest, predictions):
        plt.scatter(xTest, predictions)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.plot()
        plt.show()
