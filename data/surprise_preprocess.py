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

def create_testset(X_test, n = 1):

    rng = np.random.default_rng(123)
    hidden_ingredients = np.zeros(X_test.shape)
    n_rows = X_test.shape[0]
    X_test = X_test.to_numpy()

    for i in range(n_rows):
        
        ingredients = np.squeeze(np.array(np.nonzero(X_test[i, :])))
        hide_mask = rng.choice(ingredients, size = n, replace = False)
        X_test[i, hide_mask] = 0
        hidden_ingredients[i, hide_mask] = 1

    return X_test, hidden_ingredients    
