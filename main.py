from data.split_train_test import *
from algorithms.RandomForest import *
from pyfiglet import Figlet

def random_forest(X_train, X_val, X_test, y_train, y_val, y_test, importances=False):
    rf = RandomForest(X_train, y_train, X_test, y_test)
    if importances:
        rf.get_importances()
    rf.plotRandomForest(y_val, rf.predict(X_val))
    print(f"Accuracy of predicted outcomes: {rf.accuracy}")
    print("Evaluating Random Forest algorithm")
    rf.evaluate_RF_model(X_train, y_train)
    print("Plotting roc curves for RF model")
    rf.plot_roc_curve(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    f = Figlet(font='slant')
    print(f.renderText('DME MiniProject'))

    X_train, X_val, X_test, y_train, y_val, y_test = get_data()

    random_forest(X_train, X_val, X_test, y_train, y_val, y_test)
