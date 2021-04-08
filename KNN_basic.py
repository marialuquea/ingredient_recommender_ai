from data.surprise_preprocess import *
from surprise import KNNBasic
from surprise.model_selection import cross_validate


data_train, data_test = split_data()
data_train = surprise_transform(data_train)
algo = KNNBasic()

sim_options = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
               }
algo = KNNBasic(sim_options=sim_options)
#cross_validate(algo, data_train, measures=['RMSE', 'MAE'], cv=5, verbose=True)

sim_options = {'name': 'msd',
               'user_based': False  # compute  similarities between items
               }
algo = KNNBasic(sim_options=sim_options)
cross_validate(algo, data_train, measures=['RMSE', 'MAE'], cv=5, verbose=True)