import numpy as np
from sklearn.model_selection import train_test_split

def split_data(mode):

    # Read the data
    cuisines = np.genfromtxt('data/recipes.csv', delimiter = ',',
    skip_header = 1)
    X = cuisines[:, :-1]
    y = cuisines[:, -1]

    if mode == 'hold-out':

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, 
        test_size=0.30, random_state=123)

        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, 
        test_size=0.33, random_state=123)

        return (X_train, X_val, X_test, y_train, y_val, y_test)

def get_labels():
    class_labels = []
    with open('data/Cuisines.csv') as f:
        for line in f:
            label = line.split(',')[1]
            class_labels.append(label)
    return class_labels