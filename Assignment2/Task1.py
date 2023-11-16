from collections import defaultdict
from surprise import KNNWithMeans, Reader
from surprise import Dataset
from surprise.accuracy import mae
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')
usersNumber = data.build_full_trainset().n_users
print(f"Users number : {usersNumber}")
mae_values = []
best_k = None


def precision_recall_at_n(predictions, n=10, threshold=4):
    """Return precision and recall at n metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of relevant and recommended items in top n
        n_rel_and_rec = sum(
            (true_r >= threshold)
            for (_, true_r) in user_ratings[:n]
        )

        # Precision@n: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec / n

        # Recall@n: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec / n_rel if n_rel != 0 else 0

    return precisions, recalls


def train_test(size):
    mae_values = []  # Reset mae_values for each sparsity level

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

    print(f"Sparcity: {size}%")
    print(f"Best K for minimizing MAE: {best_k}")
    print(f"Lowest MAE: {min_mae}")

    plt.plot(k_values_to_test, mae_values, marker='o', label='MAE values')
    plt.scatter(best_k, min_mae, color='red', label=f'Best K={best_k}\nLowest MAE={min_mae}', zorder=5)
    plt.title(f'MAE values for different k (Sparsity: {size}%)')  # Updated line
    plt.xlabel('k values')
    plt.ylabel('MAE')
    plt.show()


# Sparcity of 25%
train_test(25)
# Sparcity of 75%
train_test(75)