import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import Dataset
from surprise import Reader

def get_data(y_val = True):
    df = pd.read_csv('data/recipes.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if y_val:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=123)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.30, random_state=123)
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.1, random_state=123)
        return X_train, X_test

def get_labels():
    return pd.read_csv('data/recipes.csv').iloc[:, :-1].columns

from surprise import Reader

def surprise_transform(data):
    '''Converts a recipes data frame into a surprise dataset'''

    cols = list(data.columns)
    data['recipe_id'] = data.index
    surprise_data = pd.melt(data,id_vars = ['recipe_id'], value_vars = cols,
    var_name = 'ingredient', value_name = 'rating')
    reader = Reader(rating_scale = (0, 1))
    return Dataset.load_from_df(surprise_data[['recipe_id', 'ingredient', 'rating']], reader)
