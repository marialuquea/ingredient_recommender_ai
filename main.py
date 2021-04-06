from data.split_train_test import *
from algorithms.RandomForest import *


def random_forest(X_train, X_val, X_test, y_train, y_val, y_test, importances=False):
    rf = RandomForest(X_train, y_train, X_test, y_test)
    if importances:
        rf.get_importances()
    rf.plotRandomForest(y_val, rf.predict(X_val))
    print(f"{rf.accuracy}")

if __name__ == '__main__':
    print('----STARTING DME MINIPROJECT----')

    X_train, X_val, X_test, y_train, y_val, y_test = get_data()

    random_forest(X_train, X_val, X_test, y_train, y_val, y_test)

