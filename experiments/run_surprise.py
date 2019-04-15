import argparse
from math import ceil
import pandas as pd
from surprise import Reader, Dataset, dump, SVD, accuracy
from surprise.model_selection import train_test_split
from timeit import default_timer as timer


def load_goodreads(data_path):
    return pd.read_csv(data_path, quotechar='"', skipinitialspace=True)


def split_data(ratings, test_size=0.2):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=test_size)
    return trainset, testset


def convert_to_epochs(ratings, iterations=10000):
    R = ratings.shape[0]
    U = ratings['user_id'].max()
    return ceil(U * iterations / R)


def train_svd(trainset, testset, lr_all=0.01, reg_all=0.02, n_epochs=10, n_factors=300):
    start = timer()
    algo = SVD(lr_all=lr_all, reg_all=reg_all, n_epochs=n_epochs, n_factors=n_factors)
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    time_elapsed = timer() - start
    print("Test RMSE: {}".format(rmse))
    print("Time elapsed: {}".format(time_elapsed))


def train_full(data, algo):
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    # save the trained SVD model
    dump.dump('../.tmp/svd', algo=algo)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains a Surprise SVD model")
    parser.add_argument('ratings_path', type=str, help="the ratings file")
    parser.add_argument('-l', '--learning_rate', type=int, default=0.01, help="the learning rate")
    parser.add_argument('-r', '--regularization_rate', type=int, default=0.02, help="the regularization rate")
    parser.add_argument('-n', '--num_iterations', type=int, default=10000, help="the number of cu2rec iterations")
    parser.add_argument('-f', '--factors', type=int, default=300, help="the number of factors")
    args = parser.parse_args()

    ratings = load_goodreads(args.ratings_path)
    trainset, testset = split_data(ratings)
    epochs = convert_to_epochs(ratings, iterations=args.num_iterations)
    train_svd(trainset, testset, lr_all=args.learning_rate, reg_all=args.regularization_rate,
              n_epochs=epochs, n_factors=args.factors)
