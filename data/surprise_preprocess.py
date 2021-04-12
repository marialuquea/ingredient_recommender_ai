import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import Dataset
from surprise import Reader

def split_data():
    df = pd.read_csv('data/recipes.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.1, random_state=123)

    return X_train, X_test

def surprise_transform(data):
    '''Converts a recipes data frame into a surprise dataset'''

    cols = list(data.columns)
    data['recipe_id'] = data.index
    surprise_data = pd.melt(data,id_vars = ['recipe_id'], value_vars = cols, 
    var_name = 'ingredient', value_name = 'rating')
    reader = Reader(rating_scale = (0, 1))
    return Dataset.load_from_df(surprise_data[['recipe_id', 'ingredient', 'rating']], reader)

def create_testset(n = 1):

    df = pd.read_csv('data/recipes.csv')
    X = df.iloc[:, :-1]
    rng = np.random.default_rng(123)
    n_rows = X.shape[0]
    cols = list(X.columns)
    X = X.to_numpy()

    for i in range(n_rows):
        
        ingredients = np.squeeze(np.array(np.nonzero(X[i, :])))
        hide_mask = rng.choice(ingredients, size = n, replace = False)
        X[i, hide_mask] = -1

    surprise_data = pd.DataFrame(data = X, columns=cols)
    surprise_data['recipe_id'] = surprise_data.index
    surprise_data = pd.melt(surprise_data,id_vars = ['recipe_id'], value_vars = cols, 
    var_name = 'ingredient', value_name = 'rating')
    hidden_ratings = surprise_data.loc[surprise_data['rating'] == -1]
    hidden_ratings = hidden_ratings.replace(-1, 1)
    surprise_data = surprise_data.loc[surprise_data['rating'] != -1]
    reader = Reader(rating_scale = (0, 1))
    surprise_data = Dataset.load_from_df(surprise_data[['recipe_id', 'ingredient', 'rating']], reader)
    hidden_ratings = Dataset.load_from_df(hidden_ratings[['recipe_id', 'ingredient', 'rating']], reader)

    return surprise_data, hidden_ratings   