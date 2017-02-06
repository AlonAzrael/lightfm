


import numpy as np
from sklearn.metrics import mean_squared_error
from time import time

from megalightfm import MegaLightFM
# print "MegaLightFM:", MegaLightFM
from megalightfm.datasets import fetch_movielens


## Surprise
from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf


mode = "surprise"
mode = "megalightfm"

n_folds = 1

if mode == "megalightfm":
    movielens = fetch_movielens()
    train = movielens['train']
    test = movielens['test']

    model = MegaLightFM(
        learning_rate=0.05, 
        learning_schedule='adagrad', 
        no_components=100,
        skip_loss_flag=0,
    )

    for i in xrange(n_folds):
        ## Benchmarking
        st = time()
        model.fit(train, epochs=8, num_threads=4, 
            regularization_flag=1)
        print "elapsed time:", time() - st
        ## End Benchmarking

        y_pred = model.predict(
            test.row, 
            test.col, 
            num_threads=4
        )

        y_true = test.data
        # yy = y_true - y_pred
        # yy[yy==np.nan] = 0
        # yy = np.sqrt(yy)
        # print "mae:", 1.0*np.sum(yy)/len(y_pred)
        print "rmse:", mean_squared_error(y_true, y_pred)**0.5

elif mode == "surprise" :
    data = Dataset.load_builtin('ml-100k')
    data.split(n_folds=n_folds)
    # data = data.build_full_trainset()

    # We'll use the famous SVD algorithm.
    algo = SVD()

    st = time()
    # Evaluate performances of our algorithm on the dataset.
    perf = evaluate(algo, data, measures=['RMSE'])
    print "elapsed time:", time() - st

    print_perf(perf)




