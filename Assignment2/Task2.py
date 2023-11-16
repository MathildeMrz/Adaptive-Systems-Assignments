from collections import defaultdict
from pprint import pprint

import pandas as pd
import numpy as np
from surprise import Reader, SVD
from surprise import Dataset
from surprise.accuracy import mae
from surprise.model_selection import train_test_split

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')


def train_test(size):
    # Dataset splitting in trainset and testset for size% sparsity
    trainset, testset = train_test_split(data, test_size=size / 100,
                                         random_state=22)
    # prepare user-based SVD for predicting ratings from trainset
    algo = SVD(random_state=3)
    algo.fit(trainset)

    # estimate the ratings for all the pairs (user, item) in testset
    predictionsSVD = algo.test(testset)
    current_mae = mae(predictionsSVD)

    print(f"Sparcity: {size}%")
    print(f"MAE: {current_mae}")


# Sparcity of 25%
train_test(25)
# Sparcity of 75%
train_test(75)
