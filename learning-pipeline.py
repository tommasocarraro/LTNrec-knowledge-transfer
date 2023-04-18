import os
from joblib import Parallel, delayed
import torch.nn
import numpy as np
import wandb
from ltnrec.data import DataManager
from ltnrec.loaders import TrainingDataLoader, ValDataLoaderRatings, TrainingDataLoaderLTNClassification, \
    ValDataLoaderRanking, TrainingDataLoaderLTNRegression
from ltnrec.models import MatrixFactorization, MFTrainerClassifier, MatrixFactorizationLTN, LTNTrainerMFClassifier, \
    LTNTrainerMFClassifierTransferLearning, MFTrainerRegression, LTNTrainerMFRegression, \
    LTNTrainerMFRegressionTransferLearning
from torch.optim import Adam
from ltnrec.utils import set_seed
from ltnrec.models import FocalLossPyTorch
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import time
from pathlib import Path
import pickle


def generate_report_dict(results_path):
    """
    Generate a report dictionary (mean and variance) given the results produced by the experiments.
    It saves the dictionary as JSON on the root folder where the results are saved (folder of the experiment).
    :param results_path: path where to take the results of the experiments
    :return:
    """
    # create result dict
    result_dict = {}
    # iterate over result files in the directory
    files = Path(results_path).glob('*')
    for file in files:
        if "genres" not in file.name:
            # get file metadata
            metadata = file.name.split("-")
            model_name = metadata[0]
            p = metadata[5].replace(".json", "")
            with open(file) as result_file:
                auc_dict = json.load(result_file)
            if model_name not in result_dict:
                result_dict[model_name] = {p: [auc_dict["auc"]]}
            elif p not in result_dict[model_name]:
                result_dict[model_name][p] = [auc_dict["auc"]]
            else:
                result_dict[model_name][p].append(auc_dict["auc"])

    for model in result_dict:
        for p in result_dict[model]:
            result_dict[model][p] = str(np.mean(result_dict[model][p])) + " +/- " + str(np.std(result_dict[model][p]))

    with open(os.path.join(results_path.split("/")[0], "results.json"), 'w') as fp:
        json.dump(result_dict, fp, indent=4)


def generate_report_table(report_path):
    """
    Generate a report table given a report json file.
    :param report_path: path to the report json file.
    """
    with open(report_path) as f:
        report = json.load(f)

    models = list(report.keys())
    folds = list(report[models[0]].keys())
    metrics = list(report[models[0]][folds[0]].keys())

    report_dict = {fold: {metric: [] for metric in metrics} for fold in folds}

    for model in models:
        for fold in folds:
            for metric in metrics:
                report_dict[fold][metric].append((round(float(report[model][fold][metric].split(" +/- ")[0]), 4),
                                                  round(float(report[model][fold][metric].split(" +/- ")[1]), 4)))

    max_fold_metric = {fold: {metric: max([metric_mean for metric_mean, _ in report_dict[fold][metric]])
                              for metric in metrics}
                       for fold in folds}

    table = "\\begin{table*}[ht!]\n\\centering\n\\begin{tabular}{ l | l | " + " | ".join(["c" for _ in models]) + " }\n"
    table += "Fold & Metric & " + " & ".join([model for model in models]) + "\\\\\n\\hline"
    for fold in report_dict:
        table += "\n\\multirow{%d}{*}{%d\%%}" % (len(metrics), float(fold) * 100)
        for metric in metrics:
            table += " &" + (" %s & " + " & ".join([("%.4f$_{(%.4f)}$" if mean_metric != max_fold_metric[fold][
                metric] else "\\textbf{%.4f}$_{(%.4f)}$") % (mean_metric, variance_metric) for
                                                    mean_metric, variance_metric in
                                                    report_dict[fold][metric]])) % metric + "\\\\\n"
        table += "\\hline"

    table += "\n\\end{tabular}\n\\caption{Test metrics}\n\\end{table*}"
    return table


# create global wandb api object
api = wandb.Api()
api.entity = "bmxitalia"

dataset = DataManager("./datasets")
SEED = 0
TR_BATCH_SIZE = 256
VAL_BATCH_SIZE = 256

p_values = list(range(2, 12, 2))

SWEEP_CONFIG = {
    'method': "bayes",
    'metric': {'goal': 'maximize', 'name': 'fbeta-1.0'},
    'parameters': {
        'k': {"values": [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]},  # [1, 16, 32, 64, 128, 256, 512]
        'init_std': {"values": [0.01, 0.001, 0.0001]},
        # slow learning rate allows to obtain a stable validation metric
        'lr': {"distribution": "log_uniform_values", "min": 0.0001, "max": 0.1},
        # high reg penalty allows to compensate for the large latent factor size
        'wd': {"distribution": "log_uniform_values", "min": 0.00001, "max": 0.1},
        'biased': {"values": [1, 0]},
        'alpha': {"values": [1, 0]},
        'gamma': {"values": [0, 1, 2, 3]},
        'mse_loss': {"values": [0, 1]}
    }
}

SWEEP_CONFIG_GRID = {
    'method': "bayes",
    'metric': {'goal': 'maximize', 'name': 'auc'},
    'parameters': {
        'k': {"values": [10, 50, 100, 250, 500]},  # [1, 16, 32, 64, 128, 256, 512]
        'init_std': {"value": 0.0001},
        'lr': {"values": [0.0001, 0.001, 0.01]},
        'wd': {"values": [0.00005, 0.0001, 0.001, 0.01]},
        'biased': {"value": 1},
        'alpha': {"value": 0},
        'gamma': {"value": 2},
        'mse_loss': {"value": 1}
    }
}

SWEEP_CONFIG_TRY = {
    'method': "grid",
    'metric': {'goal': 'maximize', 'name': 'auc'},
    'parameters': {
        'k': {"values": [10, 50]},  # [1, 16, 32, 64, 128, 256, 512]
        'init_std': {"value": 0.0001},
        'lr': {"value": 0.001},
        'wd': {"value": 0.0001},
        'biased': {"value": 1},
        'alpha': {"value": 0},
        'gamma': {"value": 2},
        'mse_loss': {"value": 1}
    }
}

SWEEP_CONFIG_LTN = {
    'method': "bayes",
    'metric': {'goal': 'maximize', 'name': 'fbeta-1.0'},
    'parameters': {
        'k': {"values": [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]},  # [1, 16, 32, 64, 128, 256, 512]
        'init_std': {"values": [0.01, 0.001, 0.0001]},
        'lr': {"distribution": "log_uniform_values", "min": 0.0001, "max": 0.1},
        'wd': {"distribution": "log_uniform_values", "min": 0.00001, "max": 0.1},
        'biased': {"values": [1, 0]},
        'p_sat_agg': {"values": p_values},
        'p_pos': {"values": p_values},
        'p_neg': {"values": p_values},
        'alpha': {"values": [0.05, 0.1, 0.5, 1, 2, 3, 4, 5]},
        'exp': {"values": [1, 2, 3]},
        'p': {"values": p_values},
        'mse_loss': {"value": 1}
    }
}

SWEEP_CONFIG_LTN_GRID = {
    'method': "bayes",
    'metric': {'goal': 'maximize', 'name': 'auc'},
    'parameters': {
        'k': {"values": [10, 50, 100, 250, 500]},
        'init_std': {"value": 0.0001},
        'lr': {"values": [0.0001, 0.001, 0.01]},
        'wd': {"values": [0.00005, 0.0001, 0.001, 0.01]},
        'biased': {"value": 1},
        'p_sat_agg': {"value": 2},
        'p_pos': {"value": 2},
        'p_neg': {"value": 2},
        'alpha': {"values": [0.05, 0.1, 1, 2, 3]},
        'exp': {"values": [1, 2, 3]},
        'p': {"values": [2, 4, 6]},
        'mse_loss': {"value": 1}
    }
}

SWEEP_CONFIG_LTN_TRY = {
    'method': "grid",
    'metric': {'goal': 'maximize', 'name': 'auc'},
    'parameters': {
        'k': {"values": [10, 50]},
        'init_std': {"value": 0.0001},
        'lr': {"value": 0.001},
        'wd': {"value": 0.0001},
        'biased': {"value": 1},
        'p_sat_agg': {"values": p_values},
        'p_pos': {"values": p_values},
        'p_neg': {"values": p_values},
        'alpha': {"value": 1},
        'exp': {"value": 2},
        'p': {"value": 2},
        'mse_loss': {"value": 1}
    }
}

SWEEP_CONFIG_LTN_TRANSFER = {
    'method': "bayes",
    'metric': {'goal': 'maximize', 'name': 'fbeta-1.0'},
    'parameters': {
        'k': {"values": [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]},  # [1, 16, 32, 64, 128, 256, 512]
        'init_std': {"value": 0.0001},
        'lr': {"distribution": "log_uniform_values", "min": 0.0001, "max": 0.1},
        'wd': {"distribution": "log_uniform_values", "min": 0.00001, "max": 0.1},
        'biased': {"value": 1},
        'p_sat_agg': {"values": p_values},
        'p_pos': {"values": p_values},
        'p_neg': {"values": p_values},
        'p_forall_f1': {"value": 8},  # I give more weight to the formula on negative, since it is more important
        'p_forall_f2': {"value": 6},  # less weight to the formula on positives
        'p_in_f1': {"value": 8},  # I want the exists to be very strict
        'p_in_f2': {"value": 4},
        # I want the forall to be very strict here but not too much because it has to be applied some times
        'forall_f1': {"value": 0},  # I want the exits on the first transfer formula
        'forall_f2': {"value": 1},  # I want the forall on the second transfer formula
        'binary_likes_genre': {"value": 0},  # the userXgenres matrix must be not binary in regression model
        'f2': {'values': [0, 1]},
        'mse_loss': {"value": 1},  # I want to train my model using a regression model
        'alpha': {"values": [0.05, 0.1, 0.5, 1, 2, 3, 4, 5]},
        'exp': {"values": [1, 2, 3]},
        'p': {"values": p_values},
        'alpha_lg': {"value": 3},  # I want this parameter to be very high to be very selective
        'exp_lg': {"value": 2},
        'n_genres': {"values": [None, 20, 30, 40, 50]},
        'epoch_threshold': {"value": 5}  # starting from the fifth epoch, I try to inject the knowledge
    }
}

SWEEP_CONFIG_LTN_TRANSFER_GRID = {
    'method': "bayes",
    'metric': {'goal': 'maximize', 'name': 'auc'},
    'parameters': {
        'k': {"values": [10, 50, 100, 250, 500]},  # [1, 16, 32, 64, 128, 256, 512]
        'init_std': {"value": 0.0001},
        'lr': {"values": [0.0001, 0.001, 0.01]},
        'wd': {"values": [0.00005, 0.0001, 0.001, 0.01]},
        'biased': {"value": 1},
        'p_sat_agg': {"values": [2, 4, 6]},
        'p_pos': {"value": 2},
        'p_neg': {"value": 2},
        'p_forall_f1': {"values": [2, 4, 6]},  # I give more weight to the formula on negative, since it is more important
        'p_forall_f2': {"values": [2, 4, 6]},  # less weight to the formula on positives
        'p_in_f1': {"values": [6, 8, 10]},  # I want the exists to be very strict
        'p_in_f2': {"values": [6, 8, 10]},
        # I want the forall to be very strict here but not too much because it has to be applied some times
        'forall_f1': {"value": 0},  # I want the exits on the first transfer formula
        'forall_f2': {"value": 1},  # I want the forall on the second transfer formula
        'binary_likes_genre': {"value": 0},  # the userXgenres matrix must be not binary in regression model
        'f2': {'values': [0, 1]},
        'mse_loss': {"value": 1},  # I want to train my model using a regression model
        'alpha': {"values": [0.05, 0.1, 1, 2, 3]},
        'exp': {"values": [1, 2, 3]},
        'p': {"values": [2, 4, 6]},
        'alpha_lg': {"values": [2, 3]},  # I want this parameter to be very high to be very selective
        'exp_lg': {"value": 2},
        'n_genres': {"value": None},
        'epoch_threshold': {"value": 5}  # starting from the fifth epoch, I try to inject the knowledge
    }
}

SWEEP_CONFIG_LTN_TRANSFER_TRY = {
    'method': "grid",
    'metric': {'goal': 'maximize', 'name': 'auc'},
    'parameters': {
        'k': {"values": [10, 50]},  # [1, 16, 32, 64, 128, 256, 512]
        'init_std': {"value": 0.0001},
        'lr': {"value": 0.001},
        'wd': {"value": 0.0001},
        'biased': {"value": 1},
        'p_sat_agg': {"value": 2},
        'p_pos': {"value": 2},
        'p_neg': {"value": 2},
        'p_forall_f1': {"value": 2},  # I give more weight to the formula on negative, since it is more important
        'p_forall_f2': {"value": 2},  # less weight to the formula on positives
        'p_in_f1': {"value": 8},  # I want the exists to be very strict
        'p_in_f2': {"value": 8},
        # I want the forall to be very strict here but not too much because it has to be applied some times
        'forall_f1': {"value": 0},  # I want the exits on the first transfer formula
        'forall_f2': {"value": 1},  # I want the forall on the second transfer formula
        'binary_likes_genre': {"value": 0},  # the userXgenres matrix must be not binary in regression model
        'f2': {'value': 1},
        'mse_loss': {"value": 1},  # I want to train my model using a regression model
        'alpha': {"value": 1},
        'exp': {"value": 2},
        'p': {"value": 2},
        'alpha_lg': {"value": 3},  # I want this parameter to be very high to be very selective
        'exp_lg': {"value": 2},
        'n_genres': {"value": None},
        'epoch_threshold': {"value": 5}  # starting from the fifth epoch, I try to inject the knowledge
    }
}


def increase_data_sparsity(train_set, p_new_data, seed=SEED, user_level=False):
    """
    It takes the given training set and reduces its ratings by only keeping the proportion of ratings given
    by `p_new_data`.

    :param train_set: training set that needs to be made more sparse
    :param p_new_data: percentage of ratings in the new dataset
    :param seed: seed used for randomly sampling of user-item interactions
    :param user_level: whether the sampling of ratings has to be performed at the user level. If True, `p_new_data`
    ratings of each user are randomly picked. If False, the ratings are randomly sampled independently from the user
    :return: the new sparse training set
    """
    if p_new_data == 1.00:
        return train_set
    else:
        # convert dataset to pandas
        train_set_df = pd.DataFrame(train_set["ratings"], columns=['userId', 'uri', 'sentiment'])
        if user_level:
            # take the given percentage of ratings from each user
            sampled_ids = train_set_df.groupby(by=["userId"]).apply(
                lambda x: x.sample(frac=p_new_data, random_state=seed).index
            ).explode().values
            # remove NaN indexes - when pandas is not able to sample due to small group size and high frac,
            # it returns an empty index which then becomes NaN when converted in numpy
            sampled_ids = sampled_ids[~np.isnan(sampled_ids.astype("float64"))]
            return {"ratings": train_set_df.iloc[sampled_ids].reset_index(drop=True).to_numpy()}
        else:
            # we use scikit-learn to take the ratings
            # note that stratify is set to True to maintain the distribution of positive and negative ratings
            _, train_set_df = train_test_split(train_set_df, random_state=seed, stratify=train_set_df["sentiment"],
                                               test_size=p_new_data)
            return {"ratings": train_set_df.to_numpy()}


def get_pos_prop(ratings):
    return (ratings[:, -1] == 1).sum() / len(ratings)


def likes_training(seed, train_set, val_set, n_users, n_items, config, metric, ltn=False, genres=False,
                   get_genre_matrix=False, test_loader=None, just_load=False, path=None):
    """
    It performs the training of the LikesGenre (if genres==True) or Likes (if genres==False) predicate
    using the given training and validation set. The validation set is used to perform early stopping and prevent
    overfitting.

    IF LTN is True, the predicate is learned using LTN classifier, otherwise it uses standard MF.

    If a test_loader is given, the function will perform the test of the model with the parameters saved at the best
    validation score.

    If get_genre_matrix is True, the function will compute the user X genres matrix using the predictions of the model.

    The function will return the user X genres matrix (get_genre_matrix==True), or the test
    dictionary (test_loader!=None), or None (get_genre_matrix==False, test_loader==None).

    :param seed: seed for reproducibility
    :param train_set: train set on which the tuning is performed
    :param val_set: validation set on which the tuning is evaluated
    :param n_users: number of users in the dataset
    :param n_items: number of items in the dataset
    :param config: configuration dictionary containing the hyper-parameter values to train the model
    :param metric: metric used to validate and test the model
    :param ltn: whether it has to be used LTN to perform the tuning or classic Matrix Factorization
    :param genres: whether the tuning has to be performed for the LikesGenre (genres==True) or Likes
    (genres==False) predicate
    :param get_genre_matrix: whether a user x genres pre-trained matrix has to be returned or not
    :param test_loader: test loader to test the performance of the model on the test set of the dataset. Defaults to
    None, meaning that the test phase is not performed.
    :param just_load: whether the weights for the model have to be just loaded from the file system or the model has
    to be learned from scratch
    :param path: path where to save the model every time a new best validation score is reached
    :param
    """
    # create loader for validation
    if "@" in metric or metric == "auc":
        val_loader = ValDataLoaderRanking(val_set, VAL_BATCH_SIZE)
    else:
        val_loader = ValDataLoaderRatings(val_set["ratings"], VAL_BATCH_SIZE)

    # get proportion of positive examples in the dataset

    pos_prop = get_pos_prop(train_set["ratings"])

    # define function for computing user x genre matrix

    def compute_u_g_matrix(mf, avoid_normalization=True):
        matrix = torch.matmul(mf.u_emb.weight.data, torch.t(mf.i_emb.weight.data))
        if mf.biased:
            matrix = torch.add(matrix, mf.u_bias.weight.data)
            matrix = torch.add(matrix, torch.t(mf.i_bias.weight.data))
        # apply sigmoid to predictions
        return torch.sigmoid(matrix) if not avoid_normalization else matrix

    # define function to call for performing the training for the standard Matrix Factorization

    def train_likes_standard():
        train_loader = TrainingDataLoader(train_set["ratings"], TR_BATCH_SIZE)
        mf = MatrixFactorization(n_users, n_items, config["k"], config["biased"],
                                 init_std=config["init_std"])
        optimizer = Adam(mf.parameters(), lr=config["lr"], weight_decay=config["wd"])
        if config["mse_loss"]:
            trainer = MFTrainerRegression(mf, optimizer, wandb_train=False)
        else:
            trainer = MFTrainerClassifier(mf, optimizer,
                                          loss=FocalLossPyTorch(alpha=(1. - pos_prop) if config["alpha"] else -1.,
                                                                gamma=config["gamma"]),
                                          wandb_train=False, threshold=0.5)
        if not just_load:
            trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1,
                          save_path=(
                              "likes_genre_standard.pth" if genres else "likes_standard.pth") if path is None else path)

        if get_genre_matrix:
            trainer.load_model("likes_genre_standard.pth" if path is None else path)
            # compute and return matrix
            return compute_u_g_matrix(mf, config["mse_loss"])

        if test_loader is not None:
            trainer.load_model("likes_standard.pth" if path is None else path)
            if "@" in metric:
                m = "1+random"
            elif metric == "auc":
                m = "auc"
            else:
                m = "rating-prediction"

            return trainer.test(test_loader, m, fbeta=metric if "fbeta" in metric else None)

    # define function to call for performing the training for the Matrix Factorization inside the LTN framework

    def train_likes_ltn():
        if config["mse_loss"]:
            train_loader = TrainingDataLoaderLTNRegression(train_set["ratings"], batch_size=TR_BATCH_SIZE)
        else:
            train_loader = TrainingDataLoaderLTNClassification(train_set["ratings"], batch_size=TR_BATCH_SIZE)
        mf = MatrixFactorizationLTN(n_users, n_items, config["k"], config["biased"],
                                    config["init_std"], normalize=True if not config["mse_loss"] else False)
        optimizer = Adam(mf.parameters(), lr=config["lr"], weight_decay=config["wd"])
        if config["mse_loss"]:
            trainer = LTNTrainerMFRegression(mf, optimizer, config["alpha"], config["exp"], config["p"])
        else:
            trainer = LTNTrainerMFClassifier(mf, optimizer, p_pos=config["p_pos"], p_neg=config["p_neg"],
                                             p_sat_agg=config["p_sat_agg"],
                                             wandb_train=False, threshold=0.5)
        if not just_load:
            trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1,
                          save_path=("likes_genre_ltn.pth" if genres else "likes_ltn.pth") if path is None else path)

        u_g_matrix, test_metric = None, None

        if get_genre_matrix:
            trainer.load_model("likes_genre_ltn.pth" if path is None else path)
            # compute and return matrix
            u_g_matrix = compute_u_g_matrix(mf, config["mse_loss"])

        if test_loader is not None:
            trainer.load_model("likes_ltn.pth" if path is None else path)
            if "@" in metric:
                m = "1+random"
            elif metric == "auc":
                m = "auc"
            else:
                m = "rating-prediction"

            test_metric = trainer.test(test_loader, m, fbeta=metric if "fbeta" in metric else None)

        return u_g_matrix, test_metric

    # launch the training of the LikesGenre predicate

    # set seed for training
    set_seed(seed)

    if ltn:
        out = train_likes_ltn()
    else:
        out = train_likes_standard()

    return out


def likes_tuning(seed, tune_config, train_set, val_set, n_users, n_items, metric, ltn=False, genres=False,
                 exp_name=None, file_name=None, sweep_id=None):
    """
    It performs the hyper-parameter tuning of the LikesGenre (if genre==True) or Likes (if genre==False) predicate
    using the given training and validation set.

    :param seed: seed for reproducibility
    :param tune_config: configuration for the tuning of hyper-parameters
    :param train_set: train set on which the tuning is performed
    :param val_set: validation set on which the tuning is evaluated
    :param n_users: number of users in the dataset
    :param n_items: number of items in the dataset
    :param metric: validation metric that has to be used
    :param ltn: whether it has to be used LTN to perform the tuning or classic Matrix Factorization
    :param genres: whether the tuning has to be performed for the LikesGenre (genres==True) or Likes
    (genres==False) predicate
    :param exp_name: name of experiment. It is used to log stuff on the corresponding WandB project
    :param file_name: name of the file where the best configuration has to be saved
    :param sweep_id: sweep id if ones wants to continue a sweep
    """
    # create loader for validating
    if "@" in metric or metric == "auc":
        val_loader = ValDataLoaderRanking(val_set, VAL_BATCH_SIZE)
    else:
        val_loader = ValDataLoaderRatings(val_set["ratings"], VAL_BATCH_SIZE)

    # get proportion of positive examples in the dataset

    pos_prop = get_pos_prop(train_set["ratings"])

    # define function to call for performing one run for the standard Matrix Factorization

    def tune_likes_standard():
        with wandb.init():
            k = wandb.config.k
            biased = wandb.config.biased
            lr = wandb.config.lr
            wd = wandb.config.wd
            gamma = wandb.config.gamma
            alpha = wandb.config.alpha
            init_std = wandb.config.init_std
            mse_loss = wandb.config.mse_loss

            train_loader = TrainingDataLoader(train_set["ratings"], TR_BATCH_SIZE)
            mf = MatrixFactorization(n_users, n_items, k, biased, init_std=init_std)
            optimizer = Adam(mf.parameters(), lr=lr, weight_decay=wd)
            if mse_loss:
                trainer = MFTrainerRegression(mf, optimizer, wandb_train=True)
            else:
                trainer = MFTrainerClassifier(mf, optimizer,
                                              loss=FocalLossPyTorch(alpha=(1. - pos_prop) if alpha else -1.,
                                                                    gamma=gamma),
                                              wandb_train=True, threshold=0.5)
            trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1)

    # define function to call for performing one run for the Matrix Factorization inside the LTN framework

    def tune_likes_ltn():
        with wandb.init():
            k = wandb.config.k
            biased = wandb.config.biased
            lr = wandb.config.lr
            init_std = wandb.config.init_std
            wd = wandb.config.wd
            p_sat_agg = wandb.config.p_sat_agg
            p_pos = wandb.config.p_pos
            p_neg = wandb.config.p_neg
            mse_loss = wandb.config.mse_loss
            alpha = wandb.config.alpha
            exp = wandb.config.exp
            p = wandb.config.p

            if mse_loss:
                train_loader = TrainingDataLoaderLTNRegression(train_set["ratings"], batch_size=TR_BATCH_SIZE)
            else:
                train_loader = TrainingDataLoaderLTNClassification(train_set["ratings"], batch_size=TR_BATCH_SIZE)

            mf = MatrixFactorizationLTN(n_users, n_items, k, biased, init_std,
                                        normalize=True if not mse_loss else False)
            optimizer = Adam(mf.parameters(), lr=lr, weight_decay=wd)
            if mse_loss:
                trainer = LTNTrainerMFRegression(mf, optimizer, alpha, exp, p, wandb_train=True)
            else:
                trainer = LTNTrainerMFClassifier(mf, optimizer, p_pos=p_pos, p_neg=p_neg, p_sat_agg=p_sat_agg,
                                                 wandb_train=True, threshold=0.5)
            trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1)

    # launch the WandB sweep for the LikesGenre predicate

    tune_config['metric']['name'] = metric

    if "value" in tune_config["parameters"]["mse_loss"] and tune_config["parameters"]["mse_loss"]["value"] == 1 \
            and "p_sat_agg" not in tune_config["parameters"]:
        tune_config["parameters"]["alpha"] = {"value": 0.5}
        tune_config["parameters"]["gamma"] = {"value": 2}
    elif "value" in tune_config["parameters"]["mse_loss"] and tune_config["parameters"]["mse_loss"][
        "value"] == 0 and "p_sat_agg" in tune_config["parameters"]:
        tune_config["parameters"]["alpha"] = {"value": 3}
        tune_config["parameters"]["exp"] = {"value": 2}
        tune_config["parameters"]["p"] = {"value": 2}
    elif "value" in tune_config["parameters"]["mse_loss"] and tune_config["parameters"]["mse_loss"][
        "value"] == 1 and "p_sat_agg" in tune_config["parameters"]:
        tune_config["parameters"]["p_sat_agg"] = {"value": 2}
        tune_config["parameters"]["p_pos"] = {"value": 2}
        tune_config["parameters"]["p_neg"] = {"value": 2}

    if "mse" in metric:
        tune_config['metric']['goal'] = "minimize"

    if sweep_id is None:
        tune_config["name"] = file_name
        sweep_id = wandb.sweep(sweep=tune_config, project=(("likes-genre-standard" if genres else "likes-standard")
                                                           if not ltn else (
            "likes-genre-ltn" if genres else "likes-ltn")) if exp_name is None else exp_name)
    set_seed(seed)
    wandb.agent(sweep_id, function=tune_likes_standard if not ltn else tune_likes_ltn,
                count=30 if tune_config['method'] != "grid" else None)

    if exp_name is not None and file_name is not None:
        best_config = api.sweep("%s/%s" % (exp_name, sweep_id)).best_run().config

        with open("%s/configs/%s.json" % (exp_name, file_name), "w") as outfile:
            json.dump(best_config, outfile, indent=4)


def final_model_tuning(seed, tune_config, train_set, val_set, n_users, n_items, item_genres_matrix, user_genres_matrix,
                       genre_idx_sort_by_pop=None, metric="auc",
                       sweep_id=None,
                       exp_name=None,
                       file_name=None):
    """
    It performs the hyper-parameter tuning of the final model. This model has a formula that forces the latent factors
    to produce the ground truth (target ratings), and a formula which acts as a kind of regularization for the latent
    factors. This second formula performs knowledge transfer and transfer learning. Based on some learned preferences
    about movie genres, we should be able to increase the performance of a model learnt to classify movie preferences.
    The performance should increase more when the dataset is made more challenging by increasing its sparsity.

    :param seed: seed for reproducibility
    :param tune_config: configuration for the WandB hyper-parameter tuning
    :param train_set: train set on which the tuning is performed
    :param val_set: validation set on which the tuning is evaluated
    :param user_genres_matrix: matrix containing the pre-trained preferences of users for genres
    :param genre_idx_sort_by_pop: indexes of genres of the dataset sorted by popularity
    :param metric: validation metric
    :param sweep_id: id of WandB sweep, to be used if a sweep has been interrupted and it is needed to continue it.
    :param
    """
    # create loader for validating
    if "@" in metric or metric == "auc":
        val_loader = ValDataLoaderRanking(val_set, VAL_BATCH_SIZE)
    else:
        val_loader = ValDataLoaderRatings(val_set["ratings"], VAL_BATCH_SIZE)

    # define function to call for performing one run for the Matrix Factorization inside the LTN framework

    def tune_model():
        with wandb.init():
            k = wandb.config.k
            biased = wandb.config.biased
            lr = wandb.config.lr
            init_std = wandb.config.init_std
            wd = wandb.config.wd
            p_sat_agg = wandb.config.p_sat_agg
            p_pos = wandb.config.p_pos
            p_neg = wandb.config.p_neg
            p_forall_f1 = wandb.config.p_forall_f1
            p_forall_f2 = wandb.config.p_forall_f2
            p_in_f1 = wandb.config.p_in_f1
            p_in_f2 = wandb.config.p_in_f2
            forall_f1 = wandb.config.forall_f1
            forall_f2 = wandb.config.forall_f2
            f2 = wandb.config.f2
            binary_likes_genre = wandb.config.binary_likes_genre
            mse_loss = wandb.config.mse_loss
            alpha = wandb.config.alpha
            exp = wandb.config.exp
            p = wandb.config.p
            alpha_lg = wandb.config.alpha_lg
            exp_lg = wandb.config.exp_lg
            n_genres_ = wandb.config.n_genres
            epoch_threshold = wandb.config.epoch_threshold

            if mse_loss:
                train_loader = TrainingDataLoaderLTNRegression(train_set["ratings"], non_relevant_sampling=True,
                                                               n_users=n_users, n_items=n_items,
                                                               batch_size=TR_BATCH_SIZE)
            else:
                train_loader = TrainingDataLoaderLTNClassification(train_set["ratings"], non_relevant_sampling=True,
                                                                   n_users=n_users, n_items=n_items,
                                                                   batch_size=TR_BATCH_SIZE)
            mf = MatrixFactorizationLTN(n_users, n_items, k, biased, init_std,
                                        normalize=True if not mse_loss else False)
            optimizer = Adam(mf.parameters(), lr=lr, weight_decay=wd)

            # filter user x genres matrix if requested
            if binary_likes_genre:
                filtered_u_g_matrix = torch.where(user_genres_matrix >= 0.5, 1., 0.)
            else:
                filtered_u_g_matrix = user_genres_matrix

            if mse_loss:
                trainer = LTNTrainerMFRegressionTransferLearning(mf, optimizer, u_g_matrix=filtered_u_g_matrix,
                                                                 i_g_matrix=item_genres_matrix,
                                                                 epoch_threshold=epoch_threshold,
                                                                 genre_idx=genre_idx_sort_by_pop[
                                                                           :n_genres_] if n_genres_ is not None else None,
                                                                 alpha=alpha, exp=exp,
                                                                 p=p, p_sat_agg=p_sat_agg, p_forall_f1=p_forall_f1,
                                                                 p_in_f1=p_in_f1, forall_f1=forall_f1, f2=f2,
                                                                 p_forall_f2=p_forall_f2, p_in_f2=p_in_f2,
                                                                 forall_f2=forall_f2, wandb_train=True,
                                                                 alpha_lg=alpha_lg, exp_lg=exp_lg)
            else:
                trainer = LTNTrainerMFClassifierTransferLearning(mf, optimizer,
                                                                 u_g_matrix=filtered_u_g_matrix,
                                                                 i_g_matrix=item_genres_matrix,
                                                                 epoch_threshold=epoch_threshold,
                                                                 genre_idx=genre_idx_sort_by_pop[
                                                                           :n_genres_] if n_genres_ is not None else None,
                                                                 p_pos=p_pos,
                                                                 p_neg=p_neg,
                                                                 p_sat_agg=p_sat_agg, p_forall_f1=p_forall_f1,
                                                                 p_in_f1=p_in_f1, forall_f1=forall_f1, f2=f2,
                                                                 p_forall_f2=p_forall_f2, p_in_f2=p_in_f2,
                                                                 forall_f2=forall_f2, wandb_train=True, threshold=0.5)

            trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1)

    # launch the WandB sweep for the Likes predicate with transfer learning

    tune_config['metric']['name'] = metric
    if "mse" in metric:
        tune_config['metric']['goal'] = "minimize"

    if sweep_id is None:
        tune_config["name"] = file_name
        sweep_id = wandb.sweep(sweep=tune_config, project="transfer-learning" if exp_name is None else exp_name)
    set_seed(seed)
    wandb.agent(sweep_id, function=tune_model, count=50 if tune_config['method'] != "grid" else None)

    if exp_name is not None and file_name is not None:
        best_config = api.sweep("%s/%s" % (exp_name, sweep_id)).best_run().config

        with open("%s/configs/%s.json" % (exp_name, file_name), "w") as outfile:
            json.dump(best_config, outfile, indent=4)


def final_model_training(seed, train_set, val_set, n_users, n_items, item_genres_matrix, user_genres_matrix, config,
                         metric,
                         genre_idx_sort_by_pop=None, test_loader=None, path=None):
    """
    It performs the training of the final model on the given dataset with the given configuration of hyper-parameters.

    If test_loader is not None, it tests the best model found on the validation set on the test set.

    :param seed: seed for reproducibility
    :param train_set: train set on which the training is performed
    :param val_set: validation set on which the model is evaluated
    :param user_genres_matrix: matrix containing the pre-trained preferences of users for the genres
    :param genre_idx_sort_by_pop: indexes of genres of the dataset sorted by popularity
    :param config: dictionary containing the configuration of hyper-parameter for learning the model
    :param metric: metric used to validate and test the model
    :param test_loader: test loader to test the performance of the model on the test set. Defaults to None
    :param path: path where to save the best model
    """
    # create loaders for training and validating
    if config["mse_loss"]:
        train_loader = TrainingDataLoaderLTNRegression(train_set["ratings"], non_relevant_sampling=True,
                                                       n_users=n_users, n_items=n_items, batch_size=TR_BATCH_SIZE)
    else:
        train_loader = TrainingDataLoaderLTNClassification(train_set["ratings"], non_relevant_sampling=True,
                                                           n_users=n_users, n_items=n_items, batch_size=TR_BATCH_SIZE)
    if "@" in metric or metric == "auc":
        val_loader = ValDataLoaderRanking(val_set, VAL_BATCH_SIZE)
    else:
        val_loader = ValDataLoaderRatings(val_set["ratings"], VAL_BATCH_SIZE)

    mf = MatrixFactorizationLTN(n_users, n_items, config["k"], config["biased"], config["init_std"],
                                normalize=True if not config["mse_loss"] else False)
    optimizer = Adam(mf.parameters(), lr=config["lr"], weight_decay=config["wd"])

    # filter user X genres matrix, if requested
    if config["binary_likes_genre"]:
        filtered_u_g_matrix = torch.where(user_genres_matrix >= 0.5, 1., 0.)
    else:
        filtered_u_g_matrix = user_genres_matrix

    if config["mse_loss"]:
        trainer = LTNTrainerMFRegressionTransferLearning(mf, optimizer,
                                                         u_g_matrix=filtered_u_g_matrix,
                                                         i_g_matrix=item_genres_matrix,
                                                         epoch_threshold=config["epoch_threshold"],
                                                         genre_idx=genre_idx_sort_by_pop[:config["n_genres"]] if config[
                                                                                                                     "n_genres"] is not None else None,
                                                         alpha=config["alpha"],
                                                         exp=config["exp"], p=config["p"],
                                                         p_sat_agg=config["p_sat_agg"],
                                                         p_forall_f1=config["p_forall_f1"],
                                                         p_in_f1=config["p_in_f1"], forall_f1=config["forall_f1"],
                                                         f2=config["f2"], p_forall_f2=config["p_forall_f2"],
                                                         p_in_f2=config["p_in_f2"], forall_f2=config["forall_f2"],
                                                         wandb_train=False)
    else:
        trainer = LTNTrainerMFClassifierTransferLearning(mf, optimizer,
                                                         u_g_matrix=filtered_u_g_matrix,
                                                         i_g_matrix=item_genres_matrix,
                                                         epoch_threshold=config["epoch_threshold"],
                                                         p_pos=config["p_pos"],
                                                         genre_idx=genre_idx_sort_by_pop[:config["n_genres"]] if config[
                                                                                                                     "n_genres"] is not None else None,
                                                         p_neg=config["p_neg"],
                                                         p_sat_agg=config["p_sat_agg"],
                                                         p_forall_f1=config["p_forall_f1"],
                                                         p_in_f1=config["p_in_f1"], forall_f1=config["forall_f1"],
                                                         f2=config["f2"], p_forall_f2=config["p_forall_f2"],
                                                         p_in_f2=config["p_in_f2"], forall_f2=config["forall_f2"],
                                                         wandb_train=False, threshold=0.5)
    # set seed for training
    set_seed(seed)
    trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1,
                  save_path="final_model.pth" if path is None else path)

    if test_loader is not None:
        trainer.load_model("final_model.pth" if path is None else path)
        if "@" in metric:
            m = "1+random"
        elif metric == "auc":
            m = "auc"
        else:
            m = "rating-prediction"

        return trainer.test(test_loader, m, fbeta=metric if "fbeta" in metric else None)


def experiment_run(seed, percentages, val_mode, metric, starting_seed, just_first_seed_tune, exp_name):
    # step 0: get dataset for the given seed
    data = dataset.get_mr_200k_dataset(seed=seed, val_mode=val_mode, val_mode_genres=val_mode)

    # step 1: hyper-parameter tuning for the LikesGenre predicate for the given seed
    current_genre_config = "LTN-genres-seed-%d" % seed
    if not os.path.exists("%s/results/%s.json" % (exp_name, "LTN-genres-seed-%d" % (seed,))):
        likes_tuning(seed, SWEEP_CONFIG_LTN_GRID, g_train_set, g_val_set, data["n_users"], data["n_genres"],
                     metric=metric, ltn=True, genres=True, exp_name=exp_name, file_name=current_genre_config)

    # get userXgenres matrix for the given seed
    with open("./%s/configs/%s.json" % (exp_name, current_genre_config)) as json_file:
        config_dict = json.load(json_file)

    val_loader = ValDataLoaderRanking(g_val_set, VAL_BATCH_SIZE)

    user_genres_matrix, val_auc = likes_training(seed, g_train_set, g_val_set, n_users, n_genres, config_dict,
                                                 metric=metric,
                                                 ltn=True, genres=True, get_genre_matrix=True, test_loader=val_loader,
                                                 path="%s/models/%s.pth" % (exp_name, "LTN-genres-seed-%d" % (seed,)))

    with open("%s/results/%s.json" % (exp_name, "LTN-genres-seed-%d" % (seed,)), "w") as outfile:
        json.dump(val_auc, outfile, indent=4)

    # start training procedure for the given percentage of training ratings
    Parallel(n_jobs=os.cpu_count())(
        delayed(subroutine)(seed, starting_seed, just_first_seed_tune, p, train_set_small, val_set, n_users, n_items,
                            metric, val_mode, exp_name, item_genres_matrix, user_genres_matrix, genre_idx_sort_by_pop,
                            test_set)
        for p in percentages)


def subroutine(seed, starting_seed, just_first_seed_tune, p, train_set, val_set, n_users, n_items, metric, val_mode,
               exp_name, item_genres_matrix, user_genres_matrix, genre_idx_sort_by_pop, test_set):
    # create dataset with the requested sparsity
    new_train = increase_data_sparsity(train_set, p, seed)

    # step 2: hyper-parameter tuning for the three models on the created dataset
    if not just_first_seed_tune or (just_first_seed_tune and seed == starting_seed):
        # hyper-parameter tuning is performed just for the first seed if `just_first_seed_tune` is True
        # tuning is performed for all seeds otherwise
        file_names = ["MF-movies-seed-%d-p-%.2f" % (seed, p), "LTN-movies-seed-%d-p-%.2f" % (seed, p),
                      "LTN_tl-movies-seed-%d-p-%.2f" % (seed, p)]
        configs = [SWEEP_CONFIG_GRID, SWEEP_CONFIG_LTN_GRID, SWEEP_CONFIG_LTN_TRANSFER_GRID]
        Parallel(n_jobs=os.cpu_count())(
            delayed(tuning)(seed, config, new_train, val_set, n_users, n_items, metric, exp_name, item_genres_matrix,
                            user_genres_matrix, genre_idx_sort_by_pop, file_name)
            for config, file_name in zip(configs, file_names))
        # # tuning of standard MF model
        # likes_tuning(seed, SWEEP_CONFIG_GRID, new_train, val_set, n_users, n_items, metric=metric, ltn=False,
        #              genres=False, exp_name=exp_name, file_name="MF-movies-seed-%d-p-%.2f" % (seed, p))
        #
        # # tuning of LTN model
        # likes_tuning(seed, SWEEP_CONFIG_LTN_GRID, new_train, val_set, n_users, n_items, metric=metric, ltn=True,
        #              genres=False, exp_name=exp_name, file_name="LTN-movies-seed-%d-p-%.2f" % (seed, p))
        #
        # # tuning of complete model
        # final_model_tuning(seed, SWEEP_CONFIG_LTN_TRANSFER_GRID, new_train, val_set, n_users, n_items,
        #                    item_genres_matrix, user_genres_matrix, genre_idx_sort_by_pop, metric=metric,
        #                    exp_name=exp_name,
        #                    file_name="LTN_tl-movies-seed-%d-p-%.2f" % (seed, p))

    # train the three models with the last best configuration found
    # create test loader
    if val_mode == "auc":
        test_loader = ValDataLoaderRanking(test_set, VAL_BATCH_SIZE)
    else:
        test_loader = ValDataLoaderRatings(test_set["ratings"], VAL_BATCH_SIZE)

    # here, we have to wait for grid searches to produce the config files
    if just_first_seed_tune:
        while not os.path.exists("./%s/configs/%s.json" % (exp_name,
                                                           "LTN_tl-movies-seed-%d-p-%.2f" % (starting_seed, p))):
            # the checked file is the last config file produced by the procedure
            # if the file exists, then also the other required files exist
            time.sleep(1)
    # train the standard MF model
    with open("./%s/configs/%s.json" % (exp_name,
                                        "MF-movies-seed-%d-p-%.2f" %
                                        (starting_seed if just_first_seed_tune else seed, p))) as json_file:
        config_dict = json.load(json_file)
    test_result = likes_training(seed, new_train, val_set, n_users, n_items, config_dict, metric=metric, ltn=False,
                                 genres=False,
                                 test_loader=test_loader,
                                 path="%s/models/%s.pth" % (exp_name, "MF-movies-seed-%d-p-%.2f" % (seed, p)))
    with open("%s/results/%s.json" % (exp_name, "MF-movies-seed-%d-p-%.2f" % (seed, p)), "w") as outfile:
        json.dump(test_result, outfile, indent=4)

    # train LTN model
    with open("./%s/configs/%s.json" % (exp_name,
                                        "LTN-movies-seed-%d-p-%.2f" %
                                        (starting_seed if just_first_seed_tune else seed, p))) as json_file:
        config_dict = json.load(json_file)
    test_result = likes_training(seed, new_train, val_set, n_users, n_items, config_dict, metric=metric, ltn=True,
                                 genres=False,
                                 test_loader=test_loader,
                                 path="%s/models/%s.pth" % (exp_name, "LTN-movies-seed-%d-p-%.2f" % (seed, p)))
    with open("%s/results/%s.json" % (exp_name, "LTN-movies-seed-%d-p-%.2f" % (seed, p)), "w") as outfile:
        json.dump(test_result, outfile, indent=4)

    # train complete model
    with open("./%s/configs/%s.json" % (exp_name,
                                        "LTN_tl-movies-seed-%d-p-%.2f" %
                                        (starting_seed if just_first_seed_tune else seed, p))) as json_file:
        config_dict = json.load(json_file)
    test_result = final_model_training(seed, new_train, val_set, n_users, n_items, item_genres_matrix,
                                       user_genres_matrix, genre_idx_sort_by_pop,
                                       config_dict, metric=metric,
                                       test_loader=test_loader,
                                       path="%s/models/%s.pth" % (
                                           exp_name, "LTN_tl-movies-seed-%d-p-%.2f" % (seed, p)))
    with open("%s/results/%s.json" % (exp_name, "LTN_tl-movies-seed-%d-p-%.2f" % (seed, p)), "w") as outfile:
        json.dump(test_result, outfile, indent=4)


def tuning(seed, config, train, val, n_users, n_items, metric, exp_name, item_genres_matrix, user_genres_matrix,
           genre_idx_sort_by_pop, file_name):
    if "exp_lg" in config:
        # tuning of complete model
        final_model_tuning(seed, SWEEP_CONFIG_LTN_TRANSFER_GRID, train, val, n_users, n_items,
                           item_genres_matrix, user_genres_matrix, genre_idx_sort_by_pop, metric=metric,
                           exp_name=exp_name,
                           file_name=file_name)
    elif "p" in config:
        # tuning of LTN model
        likes_tuning(seed, SWEEP_CONFIG_LTN_GRID, train, val, n_users, n_items, metric=metric, ltn=True,
                     genres=False, exp_name=exp_name, file_name=file_name)
    else:
        # tuning of standard MF model
        likes_tuning(seed, SWEEP_CONFIG_GRID, train, val, n_users, n_items, metric=metric, ltn=False,
                     genres=False, exp_name=exp_name, file_name=file_name)


def create_dataset(seed, val_mode, percentages, data_path):
    if not os.path.exists(os.path.join(data_path, str(seed))):
        print("Creating dataset with seed %d" % (seed,))
        data = dataset.get_mr_200k_dataset(seed, val_mode=val_mode, val_mode_genres=val_mode, k_filter_genre=5,
                                           k_filter_movie=5, remove_top_genres=True, remove_top_movies=True)
        data["p"] = {}
        for p in percentages:
            data["p"][p] = increase_data_sparsity(data["i_tr_small"], p, seed)
        with open(os.path.join(data_path, str(seed)), 'wb') as dataset_file:
            pickle.dump(data, dataset_file)


def grid_lg(seed, exp_name, metric):
    if not os.path.exists("%s/configs/%s.json" % (exp_name, "LTN-genres-seed-%d" % (seed,))):
        with open("%s/datasets/%d" % (exp_name, seed), 'rb') as dataset_file:
            data = pickle.load(dataset_file)
        print("Performing grid search of LikesGenre with seed %d" % (seed,))
        likes_tuning(seed, SWEEP_CONFIG_LTN_GRID, data["g_tr"], data["g_val"], data["n_users"], data["n_genres"],
                     metric=metric, ltn=True, genres=True, exp_name=exp_name, file_name="LTN-genres-seed-%d" % (seed,))

        # update dataset to include userXgenres matrix
        with open("./%s/configs/%s.json" % (exp_name, "LTN-genres-seed-%d" % (seed,))) as json_file:
            config_dict = json.load(json_file)

        # get also the best val score to understand the quality of generated embeddings
        val_loader = ValDataLoaderRanking(data["g_val"], VAL_BATCH_SIZE)

        user_genres_matrix, val_auc_genres = likes_training(seed, data["g_tr"], data["g_val"], data["n_users"],
                                                            data["n_genres"],
                                                            config_dict,
                                                            metric=metric,
                                                            ltn=True, genres=True, get_genre_matrix=True,
                                                            test_loader=val_loader,
                                                            path="%s/models/%s.pth" % (
                                                                exp_name, "LTN-genres-seed-%d" % (seed,)))

        with open("%s/results/%s.json" % (exp_name, "LTN-genres-seed-%d" % (seed,)), "w") as outfile:
            json.dump(val_auc_genres, outfile, indent=4)

        data["u_g_matrix"] = user_genres_matrix

        # overwrite existing file to include the user-genres matrix
        with open(os.path.join(exp_name, "datasets", str(seed)), 'wb') as dataset_file:
            pickle.dump(data, dataset_file)


def grid_train_likes(seed, p, model, exp_name, metric, starting_seed=None):
    # prepare dataset for grid search
    file_name_grid = "%s-movies-seed-%d-p-%.2f" % (model, seed if starting_seed is None else starting_seed, p)
    file_name = "%s-movies-seed-%d-p-%.2f" % (model, seed, p)
    with open("%s/datasets/%d" % (exp_name, seed), 'rb') as dataset_file:
        data = pickle.load(dataset_file)
    # is starting_seed is given, the grid search has to be computed just for the first seed
    if starting_seed is None or seed == starting_seed:
        # grid search is run only if the config file does not exist
        if not os.path.exists("%s/configs/%s.json" % (exp_name, file_name_grid)):
            print("Performing grid search of %s with seed %d and proportion %.2f" % (model, seed, p))
            if model == 'MF':
                likes_tuning(seed, SWEEP_CONFIG_GRID, data["p"][p], data["i_val"], data["n_users"],
                             data["n_items"],
                             metric=metric, ltn=False, genres=False, exp_name=exp_name,
                             file_name=file_name_grid)
            elif model == "LTN":
                likes_tuning(seed, SWEEP_CONFIG_LTN_GRID, data["p"][p], data["i_val"], data["n_users"],
                             data["n_items"],
                             metric=metric, ltn=True, genres=False, exp_name=exp_name,
                             file_name=file_name_grid)
            else:
                final_model_tuning(seed, SWEEP_CONFIG_LTN_TRANSFER_GRID, data["p"][p], data["i_val"],
                                   data["n_users"],
                                   data["n_items"], data["i_g_matrix"], data["u_g_matrix"], metric, exp_name=exp_name,
                                   file_name=file_name_grid)

    # after the grid search, we can train the model
    # training is done only is the result file does not exist
    if not os.path.exists("%s/results/%s.json" % (exp_name, file_name)):
        print("Performing training of %s with seed %d and proportion %.2f" % (model, seed, p))
        test_loader = ValDataLoaderRanking(data["i_test"], VAL_BATCH_SIZE)
        # fetch best config file
        config_dict = None
        # this while has to be done because we are doing stuff in parallel, so it is possible that one process arrives
        # here before the grid search is completed. This happens when the grid search has to be performed just for the
        # first seed
        while config_dict is None:
            try:
                with open("./%s/configs/%s.json" % (exp_name, file_name_grid)) as json_file:
                    config_dict = json.load(json_file)
            except FileNotFoundError:
                print("Waiting for file %s to be generated by grid search..." %
                      ("./%s/configs/%s.json" % (exp_name, file_name_grid)))
        if model == 'MF':
            # train the standard MF model
            test_result = likes_training(seed, data["p"][p], data["i_val"], data["n_users"], data["n_items"],
                                         config_dict, metric=metric, ltn=False, genres=False,
                                         test_loader=test_loader,
                                         path="%s/models/%s.pth" % (exp_name, file_name))
        elif model == 'LTN':
            # train LTN model
            _, test_result = likes_training(seed, data["p"][p], data["i_val"], data["n_users"], data["n_items"],
                                            config_dict, metric=metric, ltn=True, genres=False,
                                            test_loader=test_loader,
                                            path="%s/models/%s.pth" % (exp_name, file_name))
        else:
            # train complete model
            test_result = final_model_training(seed, data["p"][p], data["i_val"], data["n_users"], data["n_items"],
                                               data["i_g_matrix"], data["u_g_matrix"], config_dict, metric=metric,
                                               test_loader=test_loader, path="%s/models/%s.pth" % (exp_name, file_name))
        # save result
        with open("%s/results/%s.json" % (exp_name, file_name), "w") as outfile:
            json.dump(test_result, outfile, indent=4)


def run_experiment(exp_name, percentages=(1.00,), models=('MF', 'LTN', 'LTN_tl'), starting_seed=0, n_runs=5,
                   val_mode="auc", metric="auc", just_first_seed_tune=False):
    """
    This function runs the entire experiment for the paper. The experiment consists of the following steps:
    For each seed (from `starting seed` to `(n_runs + starting seed)`), and for each fold percentage:
    1. create the datasets with the given seed
    2. tune the LTN model on the genre preferences of the dataset to learn the LikesGenre predicate (maybe it is not
    necessary to tune it for each seed)
    3. train the model with the best hyper-parameters and get the userXgenre matrix, containing the learned genre
    preferences of the users
    4. tune standard MF and LTN on movie preferences (just for initial seed)
    5. tune the transfer learning model on movie preferences using the learned userXgenre matrix (just for initial seed)
    6. train the models with the best hyper-parameters and check the test performances
    7. compare the performances of the three models and save the report of each seed and model on disk

    After all these experiments, the function produces a report table with the test metrics average across the runs.

    :param exp_name: name of the experiment. A folder to save best models and configuration JSONs will be created with
    this name
    :param percentages: percentages of training ratings on the different datasets that have to be tried by the procedure
    :param models: list of model names. The procedure will be applied to these models
    :param starting_seed: the seed from which the experiments begins
    :param n_runs: number of times that the experiment has to be repeated. More times means more stability on the
    test metrics
    :param val_mode: validation mode for the creation of datasets
    :param metric: evaluation and test metric used in the experiments
    :param just_first_seed_tune: whether the tuning of the hyper-parameters for the movie models has to be performed
    just for the first seed or not

    :return: report table in string format, that can be copied and pasted in LaTeX
    """
    # creating folders for the experiments
    if not os.path.exists(exp_name):
        os.mkdir(exp_name)
        os.mkdir(os.path.join(exp_name, "datasets"))
        os.mkdir(os.path.join(exp_name, "configs"))
        os.mkdir(os.path.join(exp_name, "models"))
        os.mkdir(os.path.join(exp_name, "results"))

    create_dataset(0, "auc", (1.0, ), "ciao")

    # create datasets for the experiment
    Parallel(n_jobs=os.cpu_count())(
        delayed(create_dataset)(seed, val_mode, percentages, os.path.join(exp_name, "datasets"))
        for seed in range(starting_seed, starting_seed + n_runs))

    # perform grid searches for the LikesGenre predicate
    # todo capire se le grid search trovano veramente i parametri giusti, mi sembra di vedere troppa inconsistenza
    #  tra le 5 run fatte, nel senso che non credo che 900 rating su un dataset molto piu' grande possano fare una
    #  grande differenza nel training. Probabilmente nel training no, ma sono challenging come validation examples?
    # todo guardare che effettivamente i datasets siano diversi tra loro
    # todo SWEEP_CONFIG_LTN_GRID guardare che sia giusto
    # todo pulire un po' il codice perche' e' uno spaghetti code brutale, tipo togliere la possibilita' di mettere o
    #  aggiungere il bias e aggiungere la possibilita' di cambiare la batch size. Provare anche a vedere se ha
    #  veramente senso introdurre le regole da epoca 5 ma secondo me si
    # todo togliere il top 2% dei popular items dal test set, per vedere come il modello si comporta sulle non-trivial
    #  recommendations
    # todo secondo me sta cosa si puo' fare anche in validation perche' cosi vediamo come va il modello sui generi
    #  non-trivial

    Parallel(n_jobs=os.cpu_count())(
        delayed(grid_lg)(seed, exp_name, metric)
        for seed in range(starting_seed, starting_seed + n_runs))

    # perform grid searches for Likes predicate for each model
    Parallel(n_jobs=os.cpu_count())(
        delayed(grid_train_likes)(seed, p, m, exp_name, metric, starting_seed if just_first_seed_tune else None)
        for seed in range(starting_seed, starting_seed + n_runs)
        for p in percentages
        for m in models)

    generate_report_dict(os.path.join(exp_name, "results"))


# function to get most popular genres
def order_genres_by_pop(genre_ratings):
    return list(genre_ratings.groupby(["uri"]).size().reset_index(
        name='counts').sort_values(by=["counts"], ascending=False)["uri"])


if __name__ == "__main__":
    # todo impostare tutti i seed sulla grid search
    os.environ["WANDB_START_METHOD"] = "thread"
    # os.environ['WANDB_MODE'] = 'offline'

    # step 0: get data for the entire experiment (data for genre and movie preferences)

    run_experiment("final-experiment", n_runs=5, percentages=(0.1, 0.05, 0.01), just_first_seed_tune=False)

    # generate_report_dict(os.path.join("final-experiment", "results"))

    # n_users, n_genres, n_items, genre_folds, movie_folds, item_genres_matrix = data.get_mr_200k_dataset(seed=0,
    #                                                                                                     val_mode="auc",
    #                                                                                                     val_mode_genres="auc")
    # # movie_val_size=0.1,
    # # movie_test_size=0.2)
    #
    # genre_ratings, g_train_set, g_val_set = genre_folds.values()
    # u_i_matrix, train_set, train_set_small, val_set, test_set = movie_folds.values()
    #
    # ordered = order_genres_by_pop(genre_ratings)
    #
    # # step 1: hyper-parameter tuning of the LikesGenre predicate (on genres) (both MF and LTN models)
    #
    # # likes_tuning(0, g_train_set, g_val_set, ltn=False, genres=True, metric="auc")
    # # likes_tuning(0, SWEEP_CONFIG_LTN_GRID, g_train_set, g_val_set, ltn=True, genres=True, metric="auc")
    # # likes_tuning(0, SWEEP_CONFIG_LTN_GRID, train_set_small, val_set, ltn=True, genres=False, metric="auc")
    # # likes_tuning(g_train_set, g_val_set, ltn=False, genres=True, metric="auc")
    # # likes_tuning(g_train_set, g_val_set, ltn=True, genres=True, metric="fbeta-1.0")
    #
    # # step 1.1: prepare dictionaries with best configurations found for the two LikesGenre models
    #
    # BEST_MF_GENRE = {"biased": 1, "k": 50, "alpha": 1, "gamma": 3, "init_std": 0.0001, "lr": 0.001, "wd": 0.00001}
    #
    # BEST_LTN_GENRE = {"biased": 1, "k": 50, "init_std": 0.0001, "lr": 0.0005, "p_neg": 3, "p_pos": 4, "p_sat_agg": 3,
    #                   "wd": 0.0001}
    #
    # BEST_MF_GENRE_AUC = {"biased": 1, "k": 500, "init_std": 0.0001, "lr": 0.0005, "wd": 0.00005, "mse_loss": 1}
    #
    # BEST_LTN_GENRE_AUC = {"biased": 1, "k": 500, "init_std": 0.0001, "lr": 0.001, "wd": 0.0005, "exp": 2,
    #                       "mse_loss": 1, "p": 2, "alpha": 1, "normalized": 0}
    #
    # BEST_LTN_GENRE_AUC_new = {"biased": 1, "k": 500, "init_std": 0.0001, "lr": 0.0001, "wd": 0.0001, "exp": 2,
    #                           "mse_loss": 1, "p": 10, "alpha": 1}
    #
    # BEST_LTN_GENRE_FBETA = {"biased": 1, "k": 100, "init_std": 0.01, "lr": 0.0001, "p_neg": 4, "p_pos": 8,
    #                         "p_sat_agg": 6, "wd": 0.0001, "mse_loss": 0}
    #
    # BEST_LTN_GENRE_AUC_MSE_NORMALIZED = {"biased": 1, "k": 400, "init_std": 0.01, "lr": 0.001, "wd": 0.00001, "exp": 2,
    #                                      "mse_loss": 1, "p": 8, "alpha": 0.05}
    #
    # BEST_LTN_GENRE_AUC_not_binary = {"biased": 1, "k": 300, "init_std": 0.0001, "lr": 0.0001, "wd": 0.0001, "exp": 3,
    #                                  "mse_loss": 1, "p": 8, "alpha": 0.05}
    #
    # # step 1.2: train the two models with the found configuration and see the results
    #
    # # likes_training(g_train_set, g_val_set, BEST_MF_GENRE, ltn=False, genres=True)  # MF reaches Epoch 35 - Train loss 0.004 - Val fbeta-1.0 0.573 - Neg prec 0.553 - Pos prec 0.894 - Neg rec 0.589 - Pos rec 0.879 - Neg f 0.570 - Pos f 0.887 - tn 527 - fp 368 - fn 426 - tp 3105
    # # likes_training(g_train_set, g_val_set, BEST_LTN_GENRE, ltn=True, genres=True)  # LTN reaches Epoch 46 - Train loss 0.341 - Val fbeta-1.0 0.566 - training_overall_sat 0.659 - pos_sat 0.645 - neg_sat 0.675 - Neg prec 0.505 - Pos prec 0.901 - Neg rec 0.635 - Pos rec 0.843 - Neg f 0.563 - Pos f 0.871 - tn 568 - fp 327 - fn 556 - tp 2975
    #
    # # likes_training(g_train_set, g_val_set, BEST_MF_GENRE_AUC, ltn=False, genres=True, metric="auc")  # Epoch 20 - Train loss 0.025 - Val auc 0.822
    # # likes_training(g_train_set, g_val_set, BEST_LTN_GENRE_FBETA, ltn=True, genres=True, metric="fbeta-1.0")
    # # likes_training(g_train_set, g_val_set, BEST_LTN_GENRE_AUC, ltn=True, genres=True, metric="auc")
    # # likes_training(0, g_train_set, g_val_set, BEST_LTN_GENRE_AUC_new, ltn=True, genres=True, metric="auc")
    #
    # # todo sul modello devo mettere alpha a 3, quello che decide se ad un utente piace un genere o meno
    #
    # # step 2: hyper-parameter tuning of the Likes predicate (on movies) (both MF and LTN)
    #
    # # likes_tuning(train_set_small, val_set, ltn=False, genres=False, metric="auc")
    # # likes_tuning(train_set_small, val_set, ltn=True, genres=False, metric="auc")
    # # likes_tuning(train_set_small, val_set, ltn=False, genres=False, metric="auc")
    # # likes_tuning(train_set_small, val_set, ltn=True, genres=False, metric="auc")
    #
    # # step 2.1: prepare dictionaries with best configurations found for the two Likes models
    #
    # BEST_MF_FBETA = {"biased": 1, "k": 50, "alpha": 1, "gamma": 0, "init_std": 0.0001, "lr": 0.001, "wd": 0.00005}
    #
    # BEST_LTN_FBETA = {"biased": 1, "k": 50, "init_std": 0.0001, "lr": 0.005, "p_neg": 3, "p_pos": 3, "p_sat_agg": 4,
    #                   "wd": 0.00005}
    #
    # BEST_MF_AUC = {"biased": 1, "k": 500, "alpha": 0, "gamma": 3, "init_std": 0.0001, "lr": 0.0005, "wd": 0.00001}
    #
    # BEST_LTN_AUC = {"biased": 1, "k": 300, "init_std": 0.0001, "lr": 0.005, "p_neg": 4, "p_pos": 4, "p_sat_agg": 4,
    #                 "wd": 0.00005}
    #
    # BEST_LTN_AUC_MSE = {"biased": 1, "k": 200, "init_std": 0.01, "lr": 0.0001, "alpha": 0.1, "exp": 1, "p": 4,
    #                     "wd": 0.00001, "mse_loss": 1}
    #
    # # step 2.2: train the two models with the found configuration and see the results
    #
    # # likes_training(train_set_small, val_set, BEST_MF, ltn=False, genres=False)  # MF reaches Epoch 39 - Train loss 0.213 - Val fbeta-1.0 0.648 - Neg prec 0.603 - Pos prec 0.816 - Neg rec 0.698 - Pos rec 0.745 - Neg f 0.647 - Pos f 0.779 - tn 1938 - fp 840 - fn 1277 - tp 3732
    # # likes_training(train_set_small, val_set, BEST_LTN, ltn=True, genres=False)  # LTN reaches Epoch 5 - Train loss 0.429 - Val fbeta-1.0 0.650 - training_overall_sat 0.571 - pos_sat 0.573 - neg_sat 0.570 - Neg prec 0.607 - Pos prec 0.817 - Neg rec 0.697 - Pos rec 0.750 - Neg f 0.649 - Pos f 0.782 - tn 1937 - fp 841 - fn 1254 - tp 3755
    # # likes_training(train_set_small, val_set, BEST_MF_AUC, ltn=False, genres=False, metric="auc")  # MF reaches Epoch 59 - Train loss 0.015 - Val auc 0.776
    # # likes_training(train_set_small, val_set, BEST_LTN_AUC, ltn=True, genres=False, metric="auc")  # LTN reaches Epoch 16 - Train loss 0.389 - Val auc 0.775 - training_overall_sat 0.611 - pos_sat 0.611 - neg_sat 0.613
    # # likes_training(0, train_set_small, val_set, BEST_LTN_AUC_MSE, ltn=True, genres=False, metric="auc")
    #
    # # todo osservazione -> la metrica su LTN e' molto piu' ballerina e mi da l'idea di aver ottenuto quel punteggio solo by chance perche' ha fatto un mega salto e non ha piu' raggiunto un punteggio simile durante il training
    #
    # # step 3: hyper-parameter tuning of the complete LTN model with the best LikesGenre model pre-trained.
    # # This model will try to learn an LTN MF model with the addition of some background knowledge on the
    # # genre preferences (if a user does not like a genre, than the user should not like movies with that genre)
    #
    # # get user x genres matrix
    #
    # user_genres_matrix = likes_training(0, g_train_set, g_val_set, BEST_LTN_GENRE_AUC, metric="auc", ltn=True,
    #                                     genres=True, get_genre_matrix=True)
    #
    # train_set_20 = increase_data_sparsity(train_set_small, 0.5, SEED)
    #
    # likes_tuning(0, SWEEP_CONFIG_LTN_GRID, train_set_20, val_set, ltn=True, genres=False, metric="auc")
    # final_model_tuning(0, SWEEP_CONFIG_LTN_TRANSFER_GRID, train_set_20, val_set, user_genres_matrix, ordered,
    #                    metric="auc")
    #
    # np.d()
    #
    # # step 3.1: prepare dictionary with configuration of best hyper-parameters
    #
    # BEST_LTN_TRANSFER_FBETA = {"biased": 1, "k": 50, "init_std": 0.0001, "lr": 0.001, "p_neg": 3, "p_pos": 3,
    #                            "p_sat_agg": 1,
    #                            "wd": 0.00005, "p_forall": 4, "p_exists": 3, "binary_likes_genre": 1, "pos_transfer": 0,
    #                            "p_pos_f_in": 2, "p_pos_f_out": 2}
    #
    # BEST_LTN_TRANSFER_AUC = {"biased": 1, "k": 20, "init_std": 0.0001, "lr": 0.001, "p_neg": 2, "p_pos": 4,
    #                          "p_sat_agg": 2, "wd": 0.00005, "p_forall_f1": 4, "p_in_f1": 4, "forall_f1": 1, "f2": 1,
    #                          "p_forall_f2": 1, "p_in_f2": 1, "forall_f2": 1, "binary_likes_genre": 1}
    #
    # BEST_LTN_TRANSFER_AUC_MSE = {"biased": 1, "k": 300, "init_std": 0.0001, "lr": 0.0001,
    #                              "p_sat_agg": 4, "wd": 0.0001, "p_forall_f1": 8, "p_in_f1": 4, "forall_f1": 1, "f2": 1,
    #                              "p_forall_f2": 8, "p_in_f2": 4, "forall_f2": 1, "binary_likes_genre": 0, "mse_loss": 1,
    #                              "alpha": 2, "exp": 2, "alpha_lg": 2, "exp_lg": 2, "n_genres": 20, "p": 4}
    #
    # BEST_LTN_TRANSFER_AUC_new = {"biased": 1, "k": 200, "init_std": 0.0001, "lr": 0.0001, "alpha": 0.5, "alpha_lg": 3,
    #                              "p_sat_agg": 2, "wd": 0.00005, "p_forall_f1": 8, "p_in_f1": 8, "forall_f1": 0, "f2": 1,
    #                              "p_forall_f2": 1, "p_in_f2": 5, "forall_f2": 4, "binary_likes_genre": 0,
    #                              "epoch_threshold": 5,
    #                              "exp": 1, "exp_lg": 2, "mse_loss": 1, "n_genres": 40, "p": 8}
    #
    # # step 3.2: train the model with the best hyper-parameters
    #
    # # final_model_training(train_set_small, val_set, BEST_LTN_TRANSFER_AUC, metric="auc")  # with k=200 we have better performance but it
    # # is not a big gap, we can keep k=50 as the other models
    # # final_model_training(0, train_set_small, val_set, user_genres_matrix, ordered, BEST_LTN_TRANSFER_AUC_new,  metric="auc")
    #
    # # step 4: compare the performance of the different models on the test set
    #
    # # compare_performance_on_test()  # todo implementare il codice per il test del modello, simile a quello per la validation
    #
    # # step 5: increase sparsity of the dataset and re-train the model at step 3 with same hyper-parameters on
    # # datasets with different sparsity to see if the knowledge about the genres helps a little bit
    # # for each dataset, we need to train both the model without genres and the model with genres and compare
    # # the differences
    #
    # # create test loader
    #
    # # test_loader = ValDataLoaderRatings(test_set["ratings"], VAL_BATCH_SIZE)
    # test_loader = ValDataLoaderRanking(test_set, VAL_BATCH_SIZE)
    #
    # train_set_02 = increase_data_sparsity(train_set_small, 0.01, SEED)
    # # print("\n ----------- Begin LTN without formula --------------- \n")
    # # print(likes_training(train_set_02, val_set, BEST_LTN_AUC, metric="auc", ltn=True, genres=False,
    # #                      test_loader=test_loader))
    # # print("\n ----------- Begin LTN with formula ------------------ \n")
    # # print(final_model_training(train_set_02, val_set, BEST_LTN_TRANSFER_AUC, metric="auc", test_loader=test_loader))
    #
    # # try with tuning and see if we are able to get massive improvements or not
    # # likes_tuning(train_set_02, val_set, metric="auc", ltn=True, genres=False)
    # likes_training(0, train_set_02, val_set, BEST_LTN_AUC, metric="auc", ltn=True, genres=False)
    # final_model_training(0, train_set_02, val_set, BEST_LTN_TRANSFER_AUC_new, metric="auc")
    # # final_model_tuning(train_set_02, val_set, "auc")
    #
    # # todo ha veramente senso apprendere mille volte il likes genre? Forse si perche' dobbiamo far vedere che il seed
    # #  cambia le performance anche li. E' giusto usare piu' seed anche su quello. Ogni volta che cambia il seed,
    # #  cambiamo anche quello, se no puo' essere che va bene solo con quel seed
    # # todo in generale, sembrerebbe che il modello con LTN genres vada sia meglio che utilizzi molti meno latent per funzionare -> molto interessante
    # # todo potrebbe essere che debba mettere un alpha basso sul Sim predicate perche' se lo mette alto non riesce ad apprendere nulla se non ho una sigmoide -> BAH
    # # todo devo ricordarmi di fare anche un esperimento in cui tengo conto di meno generi per usare la regola sui generi, magari quelli piu' popolari, che sono i macro generi
    # # todo se non gli piace un genere del casso, magari un film lo ha solo perche' il dataset e' rumoroso e etichettato male, come accade nella maggior parte di wikidata con i generi
    # # todo notare che andare bene sulla AUC significa andare bene anche in MSE e ACC in un certo senso
    #
    # # todo forse per aumentare l'AUC si potrebbe provare ad aumentare il divario tra le classi usando la MSE come loss,
    # #  tipo usare 5 e -5 potrebbe essere un'idea
    #
    # # todo provare anche con mse perche' funziona meglio per ranking, perche' distacca il decision boundary
    # # todo se invece devo fare classificazione, va bene cosi e uso fbeta o acc, devo solo decidere cosa voglio fare
    # # todo anche con mse si potrebbero fare gli esperimenti. Dobbiamo vedere quale modello funziona meglio. Quindi, la
    # #  metrica di validazione puo' rimanere la stessa, pero' il modello puo' cambiare e anche la loss puo' cambiare,
    # #  vogliamo capire quale loss porta ai migliori risultati per quale metrica -> forse devo implementare prima questa
    # #  cosa, perche' e' assolutamente importante
    #
    # # todo forse, non facendolo user-level, la valutazione e' ancora piu' fair perche' togliamo bias dagli utenti
    # # todo sarebbe utile anche fare un esperimento per il cold-start - pero' quello potrebbe essere un paper successivo
    # # todo serve fare l'esperimento con piu' seed diversi che cambiano il dataset, per vedere come il modello si
    # #  comporta
    # # todo hyper-parameter tuning teniamo quella sul seed 0
    # # todo forse bisogna aggiungere anche l'esperimento che mette la formula anche sui positivi
    # # todo bisognare fare tuning o diminuire il numero di fattori latenti
    # # todo la fbeta e' meglio ma non mi spiego il perche', va bene sulla classe negativa ma male sulla positiva.
    # #  Probabilmente perche' quella formula abbassa i negativi e non fa niente per i positivi. Abbiamo pochi positivi
    # #  per generalizzare e quindi serve anche la formula positiva, magari con un per ogni
    # # todo dovrei fare tuning su ogni piccolo dataset piu' sparso, per essere sicuro che sia tutto ok
    # # todo provare a usare l'accuracy per validare il modello perche' abbiamo le classi che sono quasi bilanciate
    # # todo forse mse e f1 sono metriche piu' appropriate e piu' stabili perche' meno sensibili al sampling, su mse e f1 si pescano molti piu' esempi rispetto a uno per utente
    # # todo notare che una migliore AUC potrebbe anche significare una migliore acc o F1, perche' indovinare l'ordine signica anche capire cosa e' positivo e cosa e' negativo
    #
    # # {'fbeta-1.0': {'fbeta-1.0': 0.519609794943261, 'neg_prec': 0.3970185579555826, 'pos_prec': 0.7272151898734177, 'neg_rec': 0.7517281105990783, 'pos_rec': 0.36697540721814115, 'neg_f': 0.519609794943261, 'pos_f': 0.4877945234557418, 'tn': 2610, 'fp': 862, 'fn': 3964, 'tp': 2298, 'sensitivity': 0.36697540721814115, 'specificity': 0.7517281105990783, 'acc': 0.5042120402712142}, 'mse': 0.4957879597287857, 'rmse': 0.7041221198973838}
    #
    # # step 0: get dataset for the current seed
    # # n_users, n_genres, n_items, genre_folds, movie_folds, \
    # # item_genres_matrix = data.get_mr_200k_dataset(seed=seed, val_mode=val_mode, val_mode_genres=val_mode)
    # # genre_ratings, g_train_set, g_val_set = genre_folds.values()
    # # genre_idx_sort_by_pop = order_genres_by_pop(genre_ratings)
    # # u_i_matrix, train_set, train_set_small, val_set, test_set = movie_folds.values()
    # #
    # # # step 1: hyper-parameter tuning for the LikesGenre predicate for the current seed
    # # current_genre_config = "LTN-genres-seed-%d" % seed
    # # likes_tuning(seed, SWEEP_CONFIG_LTN_TRY, g_train_set, g_val_set, n_users, n_genres, metric=metric, ltn=True,
    # #              genres=True, exp_name=exp_name, file_name=current_genre_config)
    # #
    # # # get userXgenres matrix for the current seed
    # # # create config dict from json
    # # with open("./%s/configs/%s.json" % (exp_name, current_genre_config)) as json_file:
    # #     config_dict = json.load(json_file)
    # #
    # # user_genres_matrix = likes_training(seed, g_train_set, g_val_set, n_users, n_genres, config_dict, metric=metric,
    # #                                     ltn=True, genres=True, get_genre_matrix=True)
    # #
    # # # for each percentage of dataset, we need to perform this internal procedure
    # # for p in percentages:
    # #     # create dataset with the requested sparsity
    # #     new_train = increase_data_sparsity(train_set_small, p, seed)
    # #     # step 2: hyper-parameter tuning for the three models on the dataset
    # #     if not just_first_seed_tune or (just_first_seed_tune and seed == starting_seed):
    # #         # hyper-parameter tuning is performed just for the first seed if `just_first_seed_tune` is True
    # #         # tuning is performed for all seeds otherwise
    # #
    # #         # tuning of standard MF model
    # #         current_MF_movie_conf = "MF-movies-seed-%d-p-%.2f" % (seed, p)
    # #         likes_tuning(seed, SWEEP_CONFIG_TRY, new_train, val_set, n_users, n_items, metric=metric, ltn=False,
    # #                      genres=False, exp_name=exp_name, file_name=current_MF_movie_conf)
    # #
    # #         # tuning of LTN model
    # #         current_LTN_movie_conf = "LTN-movies-seed-%d-p-%.2f" % (seed, p)
    # #         likes_tuning(seed, SWEEP_CONFIG_LTN_TRY, new_train, val_set, n_users, n_items, metric=metric, ltn=True,
    # #                      genres=False, exp_name=exp_name, file_name=current_LTN_movie_conf)
    # #
    # #         # tuning of complete model
    # #         current_LTN_tl_movie_conf = "LTN-tl-movies-seed-%d-p-%.2f" % (seed, p)
    # #         final_model_tuning(seed, SWEEP_CONFIG_LTN_TRANSFER_TRY, new_train, val_set, n_users, n_items, item_genres_matrix, user_genres_matrix,
    # #                            genre_idx_sort_by_pop, metric=metric, exp_name=exp_name,
    # #                            file_name=current_LTN_tl_movie_conf)
    # #
    # #     # train the three models with the last best configuration found
    # #     # create test loader
    # #     if val_mode == "auc":
    # #         test_loader = ValDataLoaderRanking(test_set, VAL_BATCH_SIZE)
    # #     else:
    # #         test_loader = ValDataLoaderRatings(test_set["ratings"], VAL_BATCH_SIZE)
    # #
    # #     # train the standard MF model
    # #     with open("./%s/configs/%s.json" % (exp_name, current_MF_movie_conf)) as json_file:
    # #         config_dict = json.load(json_file)
    # #     test_result = likes_training(seed, new_train, val_set, n_users, n_items, config_dict, metric=metric, ltn=False, genres=False,
    # #                                  test_loader=test_loader,
    # #                                  path="%s/models/%s.pth" % (exp_name, "MF-movies-seed-%d-p-%.2f" % (seed, p)))
    # #     with open("%s/results/%s.json" % (exp_name, "MF-movies-seed-%d-p-%.2f" % (seed, p)), "w") as outfile:
    # #         json.dump(test_result, outfile, indent=4)
    # #
    # #     # train LTN model
    # #     with open("./%s/configs/%s.json" % (exp_name, current_LTN_movie_conf)) as json_file:
    # #         config_dict = json.load(json_file)
    # #     test_result = likes_training(seed, new_train, val_set, n_users, n_items, config_dict, metric=metric, ltn=True, genres=False,
    # #                                  test_loader=test_loader,
    # #                                  path="%s/models/%s.pth" % (exp_name, "LTN-movies-seed-%d-p-%.2f" % (seed, p)))
    # #     with open("%s/results/%s.json" % (exp_name, "LTN-movies-seed-%d-p-%.2f" % (seed, p)), "w") as outfile:
    # #         json.dump(test_result, outfile, indent=4)
    # #
    # #     # train complete model
    # #     with open("./%s/configs/%s.json" % (exp_name, current_LTN_tl_movie_conf)) as json_file:
    # #         config_dict = json.load(json_file)
    # #     test_result = final_model_training(seed, new_train, val_set, n_users, n_items, item_genres_matrix, user_genres_matrix, genre_idx_sort_by_pop,
    # #                                        config_dict, metric=metric,
    # #                                        test_loader=test_loader,
    # #                                        path="%s/models/%s.pth" % (
    # #                                            exp_name, "LTN-tl-movies-seed-%d-p-%.2f" % (seed, p)))
    # #     with open("%s/results/%s.json" % (exp_name, "LTN-tl-movies-seed-%d-p-%.2f" % (seed, p)), "w") as outfile:
    # #         json.dump(test_result, outfile, indent=4)
