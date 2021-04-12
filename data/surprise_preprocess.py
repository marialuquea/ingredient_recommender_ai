import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import Dataset
from surprise import Reader

def split_data():
    df = pd.read_csv('data/recipes.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.05, random_state=123)

    return X_train, X_test

def surprise_transform(data):
    '''Converts a recipes data frame into a surprise dataset'''

    cols = list(data.columns)
    data['recipe_id'] = data.index
    surprise_data = pd.melt(data,id_vars = ['recipe_id'], value_vars = cols, 
    var_name = 'ingredient', value_name = 'rating')
    reader = Reader(rating_scale = (0, 1))
    return Dataset.load_from_df(surprise_data[['recipe_id', 'ingredient', 'rating']], reader)

def create_recommendation_set(n = 1):

    X_train, X_test = split_data()
    cols = list(X_train.columns)
    X_train['recipe_id'] = X_train.index
    surprise_data = pd.melt(X_train,id_vars = ['recipe_id'], value_vars = cols, 
    var_name = 'ingredient', value_name = 'rating')

    rng = np.random.default_rng(123)
    n_rows = X_test.shape[0]
    cols = list(X_test.columns)
    X_test = X_test.to_numpy()

    for i in range(n_rows):
        
        ingredients = np.squeeze(np.array(np.nonzero(X_test[i, :])))
        hide_mask = rng.choice(ingredients, size = n, replace = False)
        X_test[i, hide_mask] = -1

    X_test = pd.DataFrame(data = X_test, columns=cols)
    X_test['recipe_id'] = X_test.index
    X_test = pd.melt(X_test,id_vars = ['recipe_id'], value_vars = cols, 
    var_name = 'ingredient', value_name = 'rating')
    hidden_ratings = X_test.loc[X_test['rating'] != 1]
    hidden_ratings = hidden_ratings.replace(-1, 1)
    full_test = X_test.copy()
    X_test = X_test.loc[X_test['rating'] == 1]
    surprise_data.append(X_test, ignore_index=True)
    reader = Reader(rating_scale = (0, 1))
    surprise_data = Dataset.load_from_df(surprise_data[['recipe_id', 'ingredient', 'rating']], reader)
    trainset = surprise_data.build_full_trainset()

    return trainset, hidden_ratings, full_test