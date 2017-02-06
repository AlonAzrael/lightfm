


import numpy as np
from sklearn.metrics import mean_squared_error
from time import time

from megalightfm import MegaLightFM
# print "MegaLightFM:", MegaLightFM
from megalightfm.datasets import fetch_movielens


movielens = fetch_movielens()
train = movielens['train']
test = movielens['test']


## boosting iteration
n_boost = 5
# sample_weight = None
sample_weight = train.copy()
sample_weight.data = np.ones(len(train.data), dtype=np.float32)
model_list = []
for i in xrange(n_boost):
    
    model = MegaLightFM(
        learning_rate=0.05, 
        learning_schedule='adagrad', 
        no_components=100,
        skip_loss_flag=0,
    )
    model_list.append(model)

    model.fit(
        train, epochs=8, num_threads=4, sample_weight=sample_weight, regularization_flag=0, )
    
    ## 
    # y_pred = model.predict(train.row, train.col, num_threads=4)
    # y_true = train.data
    # yy = y_true - y_pred
    # yy[yy==np.nan] = 0
    # yy = np.abs(yy)
    # sample_weight.data = yy


dm = test
y_pred = np.zeros_like(dm.data, dtype=np.float64)
for model in model_list:
    x = model.predict(dm.row, dm.col, num_threads=4)
    y_pred += x
y_pred /= len(model_list)

y_true = dm.data

print "rmse:", mean_squared_error(y_true, y_pred)**0.5










