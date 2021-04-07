import pandas as pd
from sklearn.model_selection import train_test_split

def get_data():
    df = pd.read_csv('data/recipes.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=123)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=123)

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_labels():
    return pd.read_csv('data/recipes.csv').iloc[:, :-1].columns