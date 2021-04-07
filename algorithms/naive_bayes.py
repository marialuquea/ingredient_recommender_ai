from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

class NaiveBayes:
    def __init__(self, xTrain, yTrain, xTest, yTest):
        print("Running Naive Bayes classifier...")

        nb = CategoricalNB(min_categories=xTrain.shape[1]).fit(xTrain.values, yTrain.values)
        self.nb = nb
        self.predictions = nb.predict(xTest)
        self.accuracy = accuracy_score(yTest.values, self.predictions)

    def predict(self, x):
        return self.nb.predict(x)
