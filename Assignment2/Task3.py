from collections import defaultdict
from surprise import KNNWithMeans, Reader
from surprise import Dataset
from surprise.accuracy import mae
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt
from surprise import Reader, SVD
from Task1 import precision_recall_at_n
from Task1 import usersNumber

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

mae_values = []
best_k = None


def train_test_knn(size):
    precision_values = []
    recall_values = []

    # Dataset splitting in trainset and testset for size% sparsity
    trainset, testset = train_test_split(data, test_size=size / 100,
                                         random_state=22)

    sim_options_KNN = {'name': "pearson",
                       'user_based': True  # compute similarities between users
                       }

    k_values_to_test = list(range(1, usersNumber))

    min_mae = float('inf')

    for k in k_values_to_test:
        # prepare user-based KNN for predicting ratings from trainset
        algo = KNNWithMeans(k, sim_options=sim_options_KNN, verbose=False)
        algo.fit(trainset)

        # estimate the ratings for all the pairs (user, item) in testset
        predictionsKNN = algo.test(testset)

        current_mae = mae(predictionsKNN)
        mae_values.append(current_mae)

        if current_mae < min_mae:
            min_mae = current_mae
            best_k = k

    best_algo = KNNWithMeans(best_k, sim_options=sim_options_KNN, verbose=False)
    best_algo.fit(trainset)
    predictionsKNN_best = best_algo.test(testset)

    print(f"Sparcity: {size}%")
    print(f"Best K for minimizing MAE: {best_k}")
    print(f"Lowest MAE: {min_mae}")

    n_values = range(10, 1000)

    for n in n_values:
        precisions, recalls = precision_recall_at_n(predictionsKNN_best, n=n, threshold=4)

        # Precision and recall can then be averaged over all users
        pre = sum(prec for prec in precisions.values()) / len(precisions)
        recall = sum(rec for rec in recalls.values()) / len(recalls)

        print(f"Precision: {pre}")
        print(f"Recall: {recall}")

        precision_values.append(pre)
        recall_values.append(recall)

    # Plotting precision and recall values
    plt.plot(n_values, precision_values, label='Precision')
    plt.plot(n_values, recall_values, label='Recall')

    # Adding labels and legend
    plt.xlabel('n Values')
    plt.ylabel('Score')
    plt.legend()
    plt.title('(KNN) Precision and Recall for different n (Sparsity: {size}%)')
    plt.show()


def train_test_svd(size):
    precision_values = []
    recall_values = []
    # Dataset splitting in trainset and testset for size% sparsity
    trainset, testset = train_test_split(data, test_size=size / 100,
                                         random_state=22)
    # prepare user-based SVD for predicting ratings from trainset
    algo = SVD(random_state=3)
    algo.fit(trainset)
    predictionsSVD = algo.test(testset)

    n_values = range(10, 1000)

    for n in n_values:
        precisions, recalls = precision_recall_at_n(predictionsSVD, n=5, threshold=4)
        # Precision and recall can then be averaged over all users
        pre = sum(prec for prec in precisions.values()) / len(precisions)
        recall = sum(rec for rec in recalls.values()) / len(recalls)

        print(f"Precision: {pre}")
        print(f"Recall: {recall}")

        precision_values.append(pre)
        recall_values.append(recall)

    # Plotting precision and recall values
    plt.plot(n_values, precision_values, label='Precision')
    plt.plot(n_values, recall_values, label='Recall')

    # Adding labels and legend
    plt.xlabel('n Values')
    plt.ylabel('Score')
    plt.legend()
    plt.title('(SVD) Precision and Recall for different n (Sparsity: {size}%)')
    plt.show()


# KNN
# Sparcity of 25%
train_test_knn(25)
# Sparcity of 75%
train_test_knn(75)

# SVD
# Sparcity of 25%
train_test_svd(25)
# Sparcity of 75%
train_test_svd(75)
