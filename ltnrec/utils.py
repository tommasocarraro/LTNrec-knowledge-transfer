import torch
import numpy as np
import random
from ltnrec.models import MatrixFactorization, MFTrainer, LTNTrainerMF, LTNTrainerMFGenres
from ltnrec.loaders import TrainingDataLoader, ValDataLoader, TrainingDataLoaderLTN, TrainingDataLoaderLTNGenres
from torch.optim import Adam
import json
import os


def set_seed(seed):
    """
    It sets the seed for the reproducibility of the experiments.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def append_to_result_file(file_name, experiment_name, result):
    """
    Append the result of a new experiment to the JSON file given in input.

    The JSON file contains all the evaluation types used in the experiments. For each evaluation type, it contains
    all the training sets on which the experiment has been run. For each training set, it contains all the models used
    to train.

    :param file_name: name of the JSON file where the result has to be added
    :param experiment_name: name of the experiment for which the result has to be added to the result file. The
    convention for the name is the following: evaluation_type-training_set-model_type, where
        - evaluation_type is the type of evaluation used for producing the result ('ml', 'ml \ mr', 'ml & mr')
        - training_set is the type of training set on which the model has been trained ('ml', 'ml | mr (movies)',
        'ml | mr (movies + genres)', 'ml (movies) | mr (genres)')
        - model_type is the type of model used in the experiment ('standard_mf', 'ltn_mf', 'ltn_mf_genres')
    :param result: a json file containing the results of the experiments
    """
    experiment = experiment_name.split("-")
    evaluation = experiment[0]
    train_set = experiment[1]
    model = experiment[2]
    # check if the file already exists - it means we have to update it
    if os.path.exists("./results/%s.json" % (file_name,)):
        # open json file
        with open("./results/%s.json" % (file_name,)) as json_file:
            data = json.load(json_file)
        # remove the file after it has been loaded - the procedure recreates the updated file
        os.remove("./results/%s.json" % (file_name,))
    else:
        # if the file does not exist - we create the file
        data = {}

    if evaluation not in data:
        # check if the experiment with this type of evaluation is already in the json file
        data[evaluation] = {train_set: {model: result}}
    else:
        if train_set not in data[evaluation]:
            # check if the experiment with this train set has been already included in the evaluation
            data[evaluation][train_set] = {model: result}
        else:
            if model not in data[evaluation][train_set]:
                # check if the experiment with this model has been already included in the results of the train_set
                data[evaluation][train_set][model] = result

    with open("./results/%s.json" % (file_name,), "w") as outfile:
        json.dump(data, outfile, indent=4)


def train_and_test(trainer, tr_loader, val_loader, test_loader, test_metrics, result_file_name, experiment_name):
    """
    It trains and tests a model.
    """
    trainer.train(tr_loader, val_loader, "hit@10", n_epochs=100, early=10, verbose=1,
                  save_path="./saved_models/%s.pth" % (experiment_name,))
    trainer.load_model("./saved_models/%s.pth" % (experiment_name,))
    result = trainer.test(test_loader, test_metrics)
    append_to_result_file(result_file_name, experiment_name, result)


def train_standard_mf(n_users, n_items, tr, val, test, test_metrics, seed, experiment_name, result_file_name):
    """
    It trains and tests a standard MF model.
    """
    with open("./config/standard_mf.json") as json_file:
        config = json.load(json_file)
    set_seed(seed)
    mf_model = MatrixFactorization(n_users, n_items, config.k, config.biased)
    tr_loader = TrainingDataLoader(tr, config.tr_batch_size)
    val_loader = ValDataLoader(val, config.val_batch_size)
    test_loader = ValDataLoader(test, config.val_batch_size)
    trainer = MFTrainer(mf_model, Adam(mf_model.parameters(), lr=config.lr, weight_decay=config.wd))
    train_and_test(trainer, tr_loader, val_loader, test_loader, test_metrics, result_file_name, experiment_name)


def train_ltn_mf(n_users, n_items, tr, val, test, test_metrics, seed, experiment_name, result_file_name):
    """
    It trains and tests a MF model trained using an LTN.
    """
    with open("./config/ltn_mf.json") as json_file:
        config = json.load(json_file)
    set_seed(seed)
    mf_model = MatrixFactorization(n_users, n_items, config.k, config.biased)
    tr_loader = TrainingDataLoaderLTN(tr, config.tr_batch_size)
    val_loader = ValDataLoader(val, config.val_batch_size)
    test_loader = ValDataLoader(test, config.val_batch_size)
    trainer = LTNTrainerMF(mf_model, Adam(mf_model.parameters(), lr=config.lr, weight_decay=config.wd),
                           alpha=config.alpha)
    train_and_test(trainer, tr_loader, val_loader, test_loader, test_metrics, result_file_name, experiment_name)


def train_ltn_mf_genres(n_users, n_items, n_genres, movie_genres, tr, val, test, test_metrics, seed,
                        experiment_name, result_file_name):
    """
    It trains and tests a MF model trained using an LTN which adds also a formula to reason about the
    genres of the movies.
    """
    with open("./config/ltn_mf_genres.json") as json_file:
        config = json.load(json_file)
    set_seed(seed)
    mf_model = MatrixFactorization(n_users, n_items, config.k, config.biased)
    # here, we pass n_items - n_genres because in this MF model the items include also the movie genres but
    # the loader needs to know the number of movies (items without genres)
    tr_loader = TrainingDataLoaderLTNGenres(tr, n_users, n_items - n_genres, n_genres, config.tr_batch_size)
    val_loader = ValDataLoader(val, config.val_batch_size)
    test_loader = ValDataLoader(test, config.val_batch_size)
    # also the trainer needs to know the exact number of movies (n_items - n_genres)
    trainer = LTNTrainerMFGenres(mf_model, Adam(mf_model.parameters(), lr=config.lr, weight_decay=config.wd),
                                 alpha=config.alpha, p=config.p,
                                 n_movies=n_items - n_genres, item_genres_matrix=movie_genres)
    train_and_test(trainer, tr_loader, val_loader, test_loader, test_metrics, result_file_name, experiment_name)
