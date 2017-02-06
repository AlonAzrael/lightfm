


import numpy as np
from sklearn.metrics import mean_squared_error
from time import time
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import os
from collection import defaultdict, Counter
import json
import cPickle as pickle

from megalightfm import MegaLightFM
# print "MegaLightFM:", MegaLightFM
from megalightfm.datasets import fetch_movielens


## using another dataset
dataset_path = "~/Downloads/ml-latest-small"

def proc_movies_tag(filepath):
    idm = {}
    movies_df = pd.read_csv(filepath)
    row, col, data = [], [], []

    ## assign each item a feature
    sid = 0
    for mrow in movies_df.iterrows():
        row.append(mrow["movieId"])
        col.append(sid)
        sid += 1
        data.append(1)

    for mrow in movies_df.iterrows():
        genres = mrow["genres"].split("|")
        for g in genres:
            if g not in idm:
                gid = len(idm) + sid
                idm[g] = gid
            else:
                gid = idm[g]

            row.append(mrow["movieId"])
            col.append(gid)
            data.append(1)
    
    result_filepath = os.path.join(os.path.dirname(filepath), "movie-tag.json")
    with open(result_filepath, "w") as F:
        json.dump(idm, F, indent=2)

    coo = coo_matrix(data, (row, col))
    item_features = csr_matrix(coo)
    result_filepath = os.path.join(os.path.dirname(filepath), "item_features.pkl")
    with open(result_filepath, "wb") as F:
        pickle.dump(item_features, F, -1)


def train_model():
    ratings_df = pd.read_csv(
        os.path.join(dataset_path, "ratings.csv"))
    interactions = coo_matrix(ratings_df["rating"], (ratings_df["userId"], ratings_df["movieId"]) )

    with open(os.path.join(dataset_path, "item_features.pkl"), "rb") as F:
        item_features = pickle.load(F)

    model = MegaLightFM(
        learning_rate=0.05, 
        learning_schedule='adagrad', 
        no_components=100,
        skip_loss_flag=0,
    )

    model.fit(interactions, item_features=item_features, epochs=8, num_threads=4, regularization_flag=1)

        with open(os.path.join(dataset_path, "model.pkl"), "wb") as F:
            pickle.dump(model, F, -1)


def item_community():
    pass





