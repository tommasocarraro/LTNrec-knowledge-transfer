import os
from joblib import Parallel, delayed
import torch.nn
import numpy as np
import wandb
from ltnrec.data import DataManager
from ltnrec.loaders import TrainingDataLoader, ValDataLoaderRatings, \
    ValDataLoaderRanking, TrainingDataLoaderLTNRegression
from ltnrec.models import MatrixFactorization, MatrixFactorizationLTN, \
    MFTrainerRegression, LTNTrainerMFRegression, LTNTrainerMFRegressionTransferLearning
from torch.optim import Adam
from ltnrec.utils import set_seed
import pandas as pd
from sklearn.model_selection import train_test_split
import json
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
VAL_BATCH_SIZE = 256

p_values = list(range(2, 12, 2))
k_values = [10, 50, 100, 200, 500]
lr_values = [0.0001, 0.001, 0.01]
wd_values = [0.00005, 0.0001, 0.001, 0.01]
bs_values = [32, 64, 128, 256, 512]

SWEEP_CONFIG_BAYES = {
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
        'k': {"values": k_values},
        'lr': {"values": lr_values},
        'wd': {"values": wd_values},
        'tr_batch_size': {"values": bs_values}
    }
}

SWEEP_CONFIG_LTN_BAYES = {
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
        'k': {"values": k_values},
        'lr': {"values": lr_values},
        'wd': {"values": wd_values},
        'tr_batch_size': {"values": bs_values},
        'alpha': {"values": [0.05, 0.1, 1, 2, 3]},
        'exp': {"values": [1, 2, 3]},
        'p': {"values": p_values}
    }
}


SWEEP_CONFIG_LTN_TRANSFER_BAYES = {
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
        'k': {"values": k_values},
        'lr': {"values": lr_values},
        'wd': {"values": wd_values},
        'tr_batch_size': {"values": bs_values},
        'p_sat_agg': {"values": p_values},
        'p_forall_f1': {"values": p_values},
        'p_forall_f2': {"values": p_values},
        'p_in_f1': {"values": p_values},
        'p_in_f2': {"values": p_values},
        'alpha': {"values": [0.05, 0.1, 1, 2, 3]},
        'exp': {"values": [1, 2, 3]},
        'p': {"values": p_values},
        'alpha_lg': {"values": [0.05, 0.1, 1, 2, 3]},
        'exp_lg': {"values": [1, 2, 3]}
    }
}

SWEEP_CONFIG_TRY = {
    'method': "grid",
    'metric': {'goal': 'maximize', 'name': 'auc'},
    'parameters': {
        'k': {"values": [10, 50]},
        'lr': {"value": 0.001},
        'wd': {"value": 0.0001},
        'tr_batch_size': {"value": 512}
    }
}

SWEEP_CONFIG_LTN_TRY = {
    'method': "grid",
    'metric': {'goal': 'maximize', 'name': 'auc'},
    'parameters': {
        'k': {"values": [10, 50]},
        'lr': {"value": 0.001},
        'wd': {"value": 0.0001},
        'alpha': {"value": 1},
        'exp': {"value": 2},
        'p': {"value": 2},
        'tr_batch_size': {"value": 512}
    }
}

SWEEP_CONFIG_LTN_TRANSFER_TRY = {
    'method': "grid",
    'metric': {'goal': 'maximize', 'name': 'auc'},
    'parameters': {
        'k': {"values": [10, 50]},
        'lr': {"value": 0.001},
        'wd': {"value": 0.0001},
        'p_sat_agg': {"value": 2},
        'p_forall_f1': {"value": 2},  # I give more weight to the formula on negative, since it is more important
        'p_forall_f2': {"value": 2},  # less weight to the formula on positives
        'p_in_f1': {"value": 8},  # I want the exists to be very strict
        'p_in_f2': {"value": 8},
        'alpha': {"value": 1},
        'exp': {"value": 2},
        'p': {"value": 2},
        'alpha_lg': {"value": 3},  # I want this parameter to be very high to be very selective
        'exp_lg': {"value": 2},
        'tr_batch_size': {"value": 512}
    }
}


def increase_data_sparsity(train_set, p_new_data, seed=0, user_level=False):
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


def likes_training(seed, train_set, val_set, n_users, n_items, config, metric, path, ltn=False,
                   get_genre_matrix=False, test_loader=None):
    """
    It performs the training of the LikesGenre or Likes predicate using the given training and validation set.
    The validation set is used to perform early stopping and prevent overfitting.

    IF LTN is True, the predicate is learned using the LTN framework, otherwise it uses standard MF.

    If a test_loader is given, the function will perform the test of the model with the parameters saved at the best
    validation score.

    If get_genre_matrix is True, the function will compute the user-genres matrix using the predictions of the model.

    The function will return the user-genres matrix (get_genre_matrix==True), and/or the test
    dictionary (test_loader!=None), or None (get_genre_matrix==False, test_loader==None).

    :param seed: seed for reproducibility
    :param train_set: train set on which the tuning is performed
    :param val_set: validation set on which the tuning is evaluated
    :param n_users: number of users in the dataset
    :param n_items: number of items in the dataset
    :param config: configuration dictionary containing the hyper-parameter values to train the model
    :param metric: metric used to validate and test the model
    :param ltn: whether it has to be used LTN to perform the tuning or classic Matrix Factorization
    :param get_genre_matrix: whether a user x genres pre-trained matrix has to be returned or not
    :param test_loader: test loader to test the performance of the model on the test set of the dataset. Defaults to
    None, meaning that the test phase is not performed.
    :param path: path where to save the model every time a new best validation score is reached
    :param
    """
    # create loader for validation
    if "@" in metric or metric == "auc":
        val_loader = ValDataLoaderRanking(val_set, VAL_BATCH_SIZE)
    else:
        val_loader = ValDataLoaderRatings(val_set["ratings"], VAL_BATCH_SIZE)

    # define function for computing user-genre matrix

    def compute_u_g_matrix(mf, normalize=True):
        matrix = torch.matmul(mf.u_emb.weight.data, torch.t(mf.i_emb.weight.data))
        matrix = torch.add(matrix, mf.u_bias.weight.data)
        matrix = torch.add(matrix, torch.t(mf.i_bias.weight.data))
        # apply sigmoid to predictions if normalization is requested
        return torch.sigmoid(matrix) if normalize else matrix

    # define function to call for performing the training for the standard Matrix Factorization

    def train_likes_standard():
        train_loader = TrainingDataLoader(train_set["ratings"], config["tr_batch_size"])
        mf = MatrixFactorization(n_users, n_items, config["k"])
        optimizer = Adam(mf.parameters(), lr=config["lr"], weight_decay=config["wd"])
        trainer = MFTrainerRegression(mf, optimizer, wandb_train=False)

        trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1, save_path=path)

        if test_loader is not None:
            trainer.load_model(path)
            if "@" in metric:
                m = "1+random"
            elif metric == "auc":
                m = "auc"
            else:
                m = "rating-prediction"

            return trainer.test(test_loader, m, fbeta=metric if "fbeta" in metric else None)

    # define function to call for performing the training for the Matrix Factorization inside the LTN framework

    def train_likes_ltn():
        train_loader = TrainingDataLoaderLTNRegression(train_set["ratings"], batch_size=config["tr_batch_size"])
        mf = MatrixFactorizationLTN(n_users, n_items, config["k"], normalize=False)
        optimizer = Adam(mf.parameters(), lr=config["lr"], weight_decay=config["wd"])
        trainer = LTNTrainerMFRegression(mf, optimizer, config["alpha"], config["exp"], config["p"])

        trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1, save_path=path)

        u_g_matrix, test_metric = None, None
        trainer.load_model(path)

        if get_genre_matrix:
            # compute and return matrix
            u_g_matrix = compute_u_g_matrix(mf, False)

        if test_loader is not None:
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


def likes_tuning(seed, tune_config, train_set, val_set, n_users, n_items, metric, ltn=False,
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

    # define function to call for performing one run for the standard Matrix Factorization

    def tune_likes_standard():
        with wandb.init():
            k = wandb.config.k
            lr = wandb.config.lr
            wd = wandb.config.wd
            tr_batch_size = wandb.config.tr_batch_size

            train_loader = TrainingDataLoader(train_set["ratings"], tr_batch_size)
            mf = MatrixFactorization(n_users, n_items, k)
            optimizer = Adam(mf.parameters(), lr=lr, weight_decay=wd)
            trainer = MFTrainerRegression(mf, optimizer, wandb_train=True)
            trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1)

    # define function to call for performing one run for the Matrix Factorization inside the LTN framework

    def tune_likes_ltn():
        with wandb.init():
            k = wandb.config.k
            tr_batch_size = wandb.config.tr_batch_size
            lr = wandb.config.lr
            wd = wandb.config.wd
            alpha = wandb.config.alpha
            exp = wandb.config.exp
            p = wandb.config.p

            train_loader = TrainingDataLoaderLTNRegression(train_set["ratings"], batch_size=tr_batch_size)
            mf = MatrixFactorizationLTN(n_users, n_items, k, normalize=False)
            optimizer = Adam(mf.parameters(), lr=lr, weight_decay=wd)
            trainer = LTNTrainerMFRegression(mf, optimizer, alpha, exp, p, wandb_train=True)
            trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1)

    # launch the WandB sweep for the LikesGenre predicate

    tune_config['metric']['name'] = metric

    if "mse" in metric:
        tune_config['metric']['goal'] = "minimize"

    if sweep_id is None:
        tune_config["name"] = file_name
        sweep_id = wandb.sweep(sweep=tune_config, project=exp_name)
    set_seed(seed)
    wandb.agent(sweep_id, function=tune_likes_standard if not ltn else tune_likes_ltn,
                count=50 if tune_config['method'] != "grid" else None)

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
            tr_batch_size = wandb.config.tr_batch_size
            k = wandb.config.k
            lr = wandb.config.lr
            wd = wandb.config.wd
            p_sat_agg = wandb.config.p_sat_agg
            p_forall_f1 = wandb.config.p_forall_f1
            p_forall_f2 = wandb.config.p_forall_f2
            p_in_f1 = wandb.config.p_in_f1
            p_in_f2 = wandb.config.p_in_f2
            alpha = wandb.config.alpha
            exp = wandb.config.exp
            p = wandb.config.p
            alpha_lg = wandb.config.alpha_lg
            exp_lg = wandb.config.exp_lg

            train_loader = TrainingDataLoaderLTNRegression(train_set["ratings"], non_relevant_sampling=True,
                                                           n_users=n_users, n_items=n_items,
                                                           batch_size=tr_batch_size)
            mf = MatrixFactorizationLTN(n_users, n_items, k, normalize=False)
            optimizer = Adam(mf.parameters(), lr=lr, weight_decay=wd)

            trainer = LTNTrainerMFRegressionTransferLearning(mf, optimizer, u_g_matrix=user_genres_matrix,
                                                             i_g_matrix=item_genres_matrix,
                                                             epoch_threshold=5,
                                                             genre_idx=None,
                                                             alpha=alpha, exp=exp,
                                                             p=p, p_sat_agg=p_sat_agg, p_forall_f1=p_forall_f1,
                                                             p_in_f1=p_in_f1, forall_f1=False, f2=True,
                                                             p_forall_f2=p_forall_f2, p_in_f2=p_in_f2,
                                                             forall_f2=True, wandb_train=True,
                                                             alpha_lg=alpha_lg, exp_lg=exp_lg)
            trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1)

    # launch the WandB sweep for the Likes predicate with transfer learning

    tune_config['metric']['name'] = metric
    if "mse" in metric:
        tune_config['metric']['goal'] = "minimize"

    if sweep_id is None:
        tune_config["name"] = file_name
        sweep_id = wandb.sweep(sweep=tune_config, project=exp_name)
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
    train_loader = TrainingDataLoaderLTNRegression(train_set["ratings"], non_relevant_sampling=True,
                                                   n_users=n_users, n_items=n_items, batch_size=config["tr_batch_size"])
    if "@" in metric or metric == "auc":
        val_loader = ValDataLoaderRanking(val_set, VAL_BATCH_SIZE)
    else:
        val_loader = ValDataLoaderRatings(val_set["ratings"], VAL_BATCH_SIZE)

    mf = MatrixFactorizationLTN(n_users, n_items, config["k"], normalize=False)
    optimizer = Adam(mf.parameters(), lr=config["lr"], weight_decay=config["wd"])

    trainer = LTNTrainerMFRegressionTransferLearning(mf, optimizer,
                                                     u_g_matrix=user_genres_matrix, i_g_matrix=item_genres_matrix,
                                                     epoch_threshold=5,
                                                     genre_idx=None,
                                                     alpha=config["alpha"],
                                                     exp=config["exp"], p=config["p"],
                                                     p_sat_agg=config["p_sat_agg"],
                                                     p_forall_f1=config["p_forall_f1"],
                                                     p_in_f1=config["p_in_f1"], forall_f1=False,
                                                     f2=True, p_forall_f2=config["p_forall_f2"],
                                                     p_in_f2=config["p_in_f2"], forall_f2=True,
                                                     wandb_train=False)
    # set seed for training
    set_seed(seed)
    trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1, save_path=path)

    if test_loader is not None:
        trainer.load_model(path)
        if "@" in metric:
            m = "1+random"
        elif metric == "auc":
            m = "auc"
        else:
            m = "rating-prediction"

        return trainer.test(test_loader, m, fbeta=metric if "fbeta" in metric else None)


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
                     metric=metric, ltn=True, exp_name=exp_name, file_name="LTN-genres-seed-%d" % (seed,))

        # update dataset to include userXgenres matrix
        with open("./%s/configs/%s.json" % (exp_name, "LTN-genres-seed-%d" % (seed,))) as json_file:
            config_dict = json.load(json_file)

        # get also the best val score to understand the quality of generated embeddings
        val_loader = ValDataLoaderRanking(data["g_val"], VAL_BATCH_SIZE)

        user_genres_matrix, val_auc_genres = likes_training(seed, data["g_tr"], data["g_val"], data["n_users"],
                                                            data["n_genres"],
                                                            config_dict,
                                                            metric=metric,
                                                            ltn=True, get_genre_matrix=True,
                                                            test_loader=val_loader,
                                                            path="%s/models/%s.pth" % (
                                                                exp_name, "LTN-genres-seed-%d" % (seed,)))

        with open("%s/results/%s.json" % (exp_name, "LTN-genres-seed-%d" % (seed,)), "w") as outfile:
            json.dump(val_auc_genres, outfile, indent=4)

        data["u_g_matrix"] = user_genres_matrix

        # overwrite existing file to include the user-genres matrix
        with open(os.path.join(exp_name, "datasets", str(seed)), 'wb') as dataset_file:
            pickle.dump(data, dataset_file)


def grid_likes(seed, p, model, exp_name, metric, starting_seed=None):
    # prepare dataset for grid search
    file_name_grid = "%s-movies-seed-%d-p-%.2f" % (model, seed if starting_seed is None else starting_seed, p)
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
                             metric=metric, ltn=False, exp_name=exp_name,
                             file_name=file_name_grid)
            elif model == "LTN":
                likes_tuning(seed, SWEEP_CONFIG_LTN_GRID, data["p"][p], data["i_val"], data["n_users"],
                             data["n_items"],
                             metric=metric, ltn=True, exp_name=exp_name,
                             file_name=file_name_grid)
            else:
                final_model_tuning(seed, SWEEP_CONFIG_LTN_TRANSFER_GRID, data["p"][p], data["i_val"],
                                   data["n_users"],
                                   data["n_items"], data["i_g_matrix"], data["u_g_matrix"], metric, exp_name=exp_name,
                                   file_name=file_name_grid)


def train_likes(seed, p, model, exp_name, metric, starting_seed=None):
    # prepare dataset for training
    file_name_grid = "%s-movies-seed-%d-p-%.2f" % (model, seed if starting_seed is None else starting_seed, p)
    file_name = "%s-movies-seed-%d-p-%.2f" % (model, seed, p)
    with open("%s/datasets/%d" % (exp_name, seed), 'rb') as dataset_file:
        data = pickle.load(dataset_file)
    # after the grid search, we can train the model
    # training is done only is the result file does not exist
    if not os.path.exists("%s/results/%s.json" % (exp_name, file_name)):
        print("Performing training of %s with seed %d and proportion %.2f" % (model, seed, p))
        test_loader = ValDataLoaderRanking(data["i_test"], VAL_BATCH_SIZE)
        # fetch best config file
        with open("./%s/configs/%s.json" % (exp_name, file_name_grid)) as json_file:
            config_dict = json.load(json_file)

        if model == 'MF':
            # train the standard MF model
            test_result = likes_training(seed, data["p"][p], data["i_val"], data["n_users"], data["n_items"],
                                         config_dict, metric=metric, ltn=False,
                                         test_loader=test_loader,
                                         path="%s/models/%s.pth" % (exp_name, file_name))
        elif model == 'LTN':
            # train LTN model
            _, test_result = likes_training(seed, data["p"][p], data["i_val"], data["n_users"], data["n_items"],
                                            config_dict, metric=metric, ltn=True,
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

    # create datasets for the experiment
    Parallel(n_jobs=os.cpu_count())(
        delayed(create_dataset)(seed, val_mode, percentages, os.path.join(exp_name, "datasets"))
        for seed in range(starting_seed, starting_seed + n_runs))

    # perform grid searches for the LikesGenre predicate
    Parallel(n_jobs=os.cpu_count())(
        delayed(grid_lg)(seed, exp_name, metric)
        for seed in range(starting_seed, starting_seed + n_runs))

    # perform grid searches for Likes predicate for each model
    Parallel(n_jobs=os.cpu_count())(
        delayed(grid_likes)(seed, p, m, exp_name, metric, starting_seed if just_first_seed_tune else None)
        for seed in range(starting_seed, starting_seed + n_runs)
        for p in percentages
        for m in models)

    # perform training of Likes predicate for each model
    Parallel(n_jobs=os.cpu_count())(
        delayed(train_likes)(seed, p, m, exp_name, metric, starting_seed if just_first_seed_tune else None)
        for seed in range(starting_seed, starting_seed + n_runs)
        for p in percentages
        for m in models)

    generate_report_dict(os.path.join(exp_name, "results"))


# function to get most popular genres
def order_genres_by_pop(genre_ratings):
    return list(genre_ratings.groupby(["uri"]).size().reset_index(
        name='counts').sort_values(by=["counts"], ascending=False)["uri"])


if __name__ == "__main__":
    os.environ["WANDB_START_METHOD"] = "thread"
    # os.environ['WANDB_MODE'] = 'offline'

    # step 0: get data for the entire experiment (data for genre and movie preferences)

    run_experiment("final-experiment-2", n_runs=5, percentages=(1.0, 0.5, 0.1, 0.05, 0.01), just_first_seed_tune=True)
