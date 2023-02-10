import os

import pandas as pd

from ltnrec.utils import set_seed
from ltnrec.data import DataManager
from ltnrec.models import MatrixFactorization, MFTrainer, LTNTrainerMF, LTNTrainerMFGenresNew
from ltnrec.loaders import ValDataLoaderExact, TrainingDataLoader, TrainingDataLoaderLTN, TrainingDataLoaderLTNGenresNew
from torch.optim import Adam
import wandb
import numpy as np
import json

METRIC = "ndcg@100"
SEED = 0

BEST_LTN = {
    "biased": 1,
    "k": 1,
    "lr": 0.01,
    "tr_batch_size": 512,
    "wd": 0.08
}

BEST_LTN_GENRES = {
    "biased": 1,
    "k": 32,
    "lr": 0.01,
    "tr_batch_size": 256,
    "wd": 0.05,
    "exists": 0,
    "genre_sample_size": 100,
    "p": 10
}


def trial_ltn():
    with wandb.init():
        k = wandb.config.k
        biased = wandb.config.biased
        lr = wandb.config.lr
        wd = wandb.config.wd
        tr_batch_size = wandb.config.tr_batch_size
        train_loader = TrainingDataLoaderLTN(train_set["ratings"], tr_batch_size)
        mf = MatrixFactorization(n_users, n_items, k, biased)
        trainer = LTNTrainerMF(mf, Adam(mf.parameters(), lr=lr, weight_decay=wd), alpha=0.05)
        trainer.train(train_loader, val_loader, METRIC, n_epochs=200, early=10, verbose=1, wandb_train=True)


def trial_ltn_genres():
    with wandb.init():
        k = wandb.config.k
        biased = wandb.config.biased
        lr = wandb.config.lr
        wd = wandb.config.wd
        tr_batch_size = wandb.config.tr_batch_size
        genre_sample_size = wandb.config.genre_sample_size
        p = wandb.config.p
        exists = wandb.config.exists
        train_loader = TrainingDataLoaderLTNGenresNew(train_set["ratings"], n_users, n_items, n_genres,
                                                      genre_sample_size, tr_batch_size)
        mf = MatrixFactorization(n_users, n_items, k, biased)
        trainer = LTNTrainerMFGenresNew(mf, Adam(mf.parameters(), lr=lr, weight_decay=wd), 0.05, p,
                                        item_genres_matrix, likes_genre_model, "./new/model_ltn.pth", exists)
        trainer.train(train_loader, val_loader, METRIC, n_epochs=200, early=10, verbose=1, wandb_train=True)


def train_ltn(config, path, train_loader):
    mf = MatrixFactorization(n_users, n_items, config["k"], config["biased"])
    trainer = LTNTrainerMF(mf, Adam(mf.parameters(), lr=config["lr"], weight_decay=config["wd"]), alpha=0.05)
    if not os.path.exists(path):
        trainer.train(train_loader, val_loader, METRIC, n_epochs=200, early=10, verbose=1, save_path=path)
    trainer.load_model(path)
    with open(path.replace(".pth", ".json"), "w") as outfile:
        json.dump(trainer.test(te_loader, ["ndcg@100", "hit@100"]), outfile, indent=4)


def train_ltn_genres(config, path, train_loader):
    mf = MatrixFactorization(n_users, n_items, config["k"], config["biased"])
    trainer = LTNTrainerMFGenresNew(mf, Adam(mf.parameters(), lr=config["lr"], weight_decay=config["wd"]), 0.05,
                                    config["p"], item_genres_matrix, likes_genre_model, "./new/model_ltn.pth",
                                    config["exists"])
    if not os.path.exists(path):
        trainer.train(train_loader, val_loader, METRIC, n_epochs=200, early=10, verbose=1,
                      save_path=path)
    trainer.load_model(path)
    with open(path.replace(".pth", ".json"), "w") as outfile:
        json.dump(trainer.test(te_loader, ["ndcg@100", "hit@100"]), outfile, indent=4)


if __name__ == "__main__":
    # connect to wandb
    wandb.login()
    # set seed for experiments
    set_seed(0)
    # generate datasets
    data = DataManager("./datasets")
    # each fold is composed of 4 datasets (train set complete (to be used with test),
    # train set small (to be used with validation), val set and test set)
    n_users, n_items, n_genres, item_folds, genre_folds, item_genres_matrix = data.get_mr_genre_ratings(SEED)
    train_set_complete, train_set, val_set, test_set = item_folds.values()
    val_loader = ValDataLoaderExact(np.unique(val_set["ratings"][:, 0]), train_set["matrix"], val_set["matrix"],
                                    batch_size=256)
    te_loader = ValDataLoaderExact(np.unique(test_set["ratings"][:, 0]), train_set_complete["matrix"],
                                   test_set["matrix"], batch_size=256)

    configuration = {
        'method': "bayes",
        'metric': {'goal': 'maximize', 'name': METRIC},
        'parameters': {
            'k': {"values": [1, 8, 16, 32, 64, 128, 256]},
            'lr': {"distribution": "log_uniform_values",
                   "min": 0.0001,
                   "max": 0.1},
            'wd': {"distribution": "log_uniform_values",
                   "min": 0.0001,
                   "max": 0.1},
            'biased': {"values": [0, 1]},
            'tr_batch_size': {"values": [64, 128, 256, 512]}}
    }

    configuration_ltn_genres = {
        'method': "bayes",
        'metric': {'goal': 'maximize', 'name': METRIC},
        'parameters': {
            'k': {"values": [1, 8, 16, 32, 64, 128, 256]},
            'lr': {"distribution": "log_uniform_values",
                   "min": 0.0001,
                   "max": 0.1},
            'wd': {"distribution": "log_uniform_values",
                   "min": 0.0001,
                   "max": 0.1},
            'biased': {"values": [0, 1]},
            'exists': {"values": [0, 1]},
            'genre_sample_size': {"values": [1, 5, 10, 50, 100, n_genres]},
            'p': {"values": [2, 5, 10]},
            'tr_batch_size': {"values": [64, 128, 256, 512]}}
    }

    # construct LikesGenre model
    likes_genre_model = MatrixFactorization(n_users, n_genres, n_factors=64, biased=1)

    # sweep_id = wandb.sweep(sweep=configuration_ltn_genres, project="best_mr_new")
    # wandb.agent(sweep_id, function=trial_ltn_genres, count=50)

    # train_ltn_genres(BEST_LTN_GENRES)

    # fatto l'ennesimo esperimento fallimentare, ora l'obiettivo e' quello di aumentare la sparsita' del dataset e vedere
    # come se la cava il nostro modello, se e' efficace o meno

    def make_sparser(prop, data, seed):
        """
        It makes the given dataset more sparse by randomly keeping just the percentage of ratings expressed by the
        parameter `prop` for each user.

        :param prop: percentage of ratings that have to be kept for each user
        :param data: dataset from which the ratings have to be removed
        :param seed: seed for random reproducibility
        :return: np.array containing the new sparse user-item rating triples
        """
        data = pd.DataFrame.from_records({"userId": data[:, 0], "uri": data[:, 1], "sentiment": data[:, 2]})
        to_remove = data.groupby(by=["userId"]).apply(
            lambda x: x.sample(frac=(1 - prop), random_state=seed).index
        ).explode().values
        # remove NaN indexes - when pandas is not able to sample due to small group size and high frac, it returns
        # an empty index which then becomes NaN when converted in numpy
        to_remove = to_remove[~np.isnan(to_remove.astype("float64"))]
        to_keep = np.setdiff1d(data.index.values, to_remove)
        new_data = data.iloc[to_keep].reset_index(drop=True)
        return np.array([tuple(rating.values()) for rating in new_data.to_dict("records")])

    props = [0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
    datasets = [make_sparser(prop, train_set["ratings"], SEED) for prop in props]

    for data, prop in zip(datasets, props):
        print("Total number of ratings: %d" % (len(train_set["ratings"]), ))
        print("Percentage of ratings in the sparse fold: %.2f" % (prop, ))
        print("Number of ratings in the sparse fold: %d" % (len(data), ))
        train_ltn(BEST_LTN, "./new/ltn_movies_%.3f.pth" % (prop, ),
                  TrainingDataLoaderLTN(data, BEST_LTN["tr_batch_size"]))
        train_ltn_genres(BEST_LTN_GENRES, "./new/ltn_new_%.3f.pth" % (prop, ),
                         TrainingDataLoaderLTNGenresNew(data, n_users, n_items, n_genres,
                                                        BEST_LTN_GENRES["genre_sample_size"],
                                                        BEST_LTN_GENRES["tr_batch_size"]))
