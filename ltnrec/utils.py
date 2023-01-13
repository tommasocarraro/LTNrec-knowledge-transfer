import copy

import torch
import numpy as np
import random
# from ltnrec.models import StandardMFModel, LTNMFModel, LTNMFGenresModel
# from ltnrec.loaders import TrainingDataLoader, ValDataLoader, TrainingDataLoaderLTN, TrainingDataLoaderLTNGenres
# from ltnrec.data import DataManager, DatasetWithGenres
# from torch.optim import Adam
import json
import os
import wandb


def set_seed(seed):
    """
    It sets the seed for the reproducibility of the experiments.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def remove_prop_and_seed_from_dataset_name(dataset_name):
    """
    It remove the proportion of training ratings and seed from the dataset name.

    :param dataset_name: name of the dataset
    :return:
    """
    name = dataset_name.split("-")
    if len(name) == 5:
        # we remove the proportion only if there is
        name = name[:-2] + [name[-1]]
    name = "-".join(name)
    return "_".join(name.split("_")[:-1]) + "_"


def remove_seed_from_dataset_name(dataset_name):
    """
    It remove the seed from the dataset name.

    :param dataset_name: name of the dataset
    :return:
    """
    return "_".join(dataset_name.split("_")[:-1]) + "_"


def reset_wandb_env():
    """
    It resets the wandb enviromet variables.

    It has to be used when multiple wandb.init() are used inside the same function. You have just to call this function
    after you run is finished. Then, you can start the next run.
    """
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def append_to_result_file(file_name, experiment_name, result, seed):
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
    :param seed: seed with which the result has been obtained. It is used to construct a directory
    """
    if not os.path.exists("./results/seed_%d" % seed):
        os.mkdir("./results/seed_%d" % seed)
    experiment = experiment_name.split("-")
    evaluation = experiment[0]
    train_set = experiment[1]
    model = experiment[2]
    # check if the file already exists - it means we have to update it
    if os.path.exists("./results/seed_%d/%s.json" % (seed, file_name)):
        # open json file
        with open("./results/seed_%d/%s.json" % (seed, file_name)) as json_file:
            data = json.load(json_file)
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

    # rewrite json file with updated information
    with open("./results/seed_%d/%s.json" % (seed, file_name), "w") as outfile:
        json.dump(data, outfile, indent=4)


def create_report_json(local_result_path_prefix, json_file_name="experiment"):
    # create dictionary containing the results
    result_dict = {}
    # iterate over created result files
    for result_file in os.listdir(local_result_path_prefix):
        # check that the file is a JSON file and begins with 'result-'
        if result_file.endswith(".json") and result_file.startswith("result-"):
            # get file content
            with open(os.path.join(local_result_path_prefix, result_file), "r") as json_file:
                metrics_values = json.load(json_file)
            # get model of the experiment
            model = result_file.split("-")[1].split("=")[1]
            # get evaluation mode of the experiment
            evaluation_mode = result_file.split("-")[2].split("=")[1]
            # get training fold of the experiment
            training_fold = result_file.split("-")[3]

            # put results at the right place in the big report dict
            if evaluation_mode not in result_dict:
                result_dict[evaluation_mode] = {training_fold: {model: {metric: [metrics_values[metric]]
                                                                        for metric in metrics_values}}}
            elif training_fold not in result_dict[evaluation_mode]:
                result_dict[evaluation_mode][training_fold] = {model: {metric: [metrics_values[metric]]
                                                                       for metric in metrics_values}}
            elif model not in result_dict[evaluation_mode][training_fold]:
                result_dict[evaluation_mode][training_fold][model] = {metric: [metrics_values[metric]]
                                                                      for metric in metrics_values}
            else:
                for metric in metrics_values:
                    result_dict[evaluation_mode][training_fold][model][metric].append(metrics_values[metric])

    # loop over the big report dict to compute mean and std of the experiments with different seeds
    for evaluation_mode in result_dict:
        for training_fold in result_dict[evaluation_mode]:
            for model in result_dict[evaluation_mode][training_fold]:
                for metric in result_dict[evaluation_mode][training_fold][model]:
                    result_dict[evaluation_mode][training_fold][model][metric] = (
                        np.mean(result_dict[evaluation_mode][training_fold][model][metric]),
                        np.std(result_dict[evaluation_mode][training_fold][model][metric]))

    # create JSON file with the result
    with open(os.path.join(local_result_path_prefix, json_file_name + ".json"), "w") as outfile:
        json.dump(result_dict, outfile, indent=4)


def create_report_dict(starting_seed=0, n_runs=30, exp_name="experiment"):
    # create dictionary containing the results
    result_dict = {}
    for seed in range(starting_seed, starting_seed + n_runs):
        assert os.path.exists("./results/seed_%d/%s.json" % (seed, exp_name)), "There is not folder containing the " \
                                                                               "results obtained with " \
                                                                               "seed %d" % (seed,)
        # get file with results for the current seed
        with open("./results/seed_%d/%s.json" % (seed, exp_name)) as json_file:
            data = json.load(json_file)
        for evaluation_mode in data:
            if evaluation_mode not in result_dict:
                result_dict[evaluation_mode] = {}
            for training_fold in data[evaluation_mode]:
                if training_fold not in result_dict[evaluation_mode]:
                    result_dict[evaluation_mode][training_fold] = {}
                for model in data[evaluation_mode][training_fold]:
                    if model not in result_dict[evaluation_mode][training_fold]:
                        result_dict[evaluation_mode][training_fold][model] = {}
                    for metric in data[evaluation_mode][training_fold][model]:
                        if metric not in result_dict[evaluation_mode][training_fold][model]:
                            result_dict[evaluation_mode][training_fold][model][metric] = \
                                [data[evaluation_mode][training_fold][model][metric]]
                        else:
                            result_dict[evaluation_mode][training_fold][model][metric].append(
                                data[evaluation_mode][training_fold][model][metric])

    # repeat the loop to compute mean and std after the results from each seed have been fetched
    for evaluation_mode in result_dict:
        for training_fold in result_dict[evaluation_mode]:
            for model in result_dict[evaluation_mode][training_fold]:
                for metric in result_dict[evaluation_mode][training_fold][model]:
                    result_dict[evaluation_mode][training_fold][model][metric] = (
                        np.mean(result_dict[evaluation_mode][training_fold][model][metric]),
                        np.std(result_dict[evaluation_mode][training_fold][model][metric]))

    return result_dict


def generate_report_table(report_dict,
                          metrics=("hit@10", "ndcg@10"),
                          models=("standard_mf", "ltn_mf", "ltn_mf_genres"),
                          evaluation_modes=("ml", "ml&mr", "ml\mr"),
                          training_folds=("ml", "ml|mr(movies)", "ml|mr(movies+genres)", "ml(movies)|mr(genres)")):
    # check if report_dict is a dict - if it is not a dict, it has to be a path to a JSON file
    if not isinstance(report_dict, dict):
        with open(report_dict, "r") as json_file:
            report_dict = json.load(json_file)

    table = "\\begin{table*}[ht!]\n"
    table += "\caption{Table title}\label{table-label}\n"
    table += "\centering\n"
    table += "\\begin{tabular}{| c | c | %s}\n" % ("".join(["c " * len(metrics) + "| " for _ in models], ))
    table += "\hline\n"
    table += "&"
    for model in models:
        table += " & \multicolumn{%d}{c |}{%s}" % (len(metrics), model.replace("_", "\\_"))
    table += "\\\\\n"
    table += "\hline\n"
    table += "&"
    for _ in models:
        for metric in metrics:
            table += " & %s" % (metric, )
    table += "\\\\\n"
    table += "\hline\n"
    for mode in evaluation_modes:
        table += "\multirow{%d}{*}{%s}" % (len(training_folds), mode.replace("\\", " $\setminus$ ").replace("&", " $\cap$ "))
        for tr_fold in training_folds:
            table += " & %s" % (tr_fold.replace("|", " $\cup$ "), )
            for model in models:
                if tr_fold in report_dict[mode] and model in report_dict[mode][tr_fold]:
                    for metric in metrics:
                        table += " & %.3f$_{(%.3f)}$" % (report_dict[mode][tr_fold][model][metric][0],
                                                         report_dict[mode][tr_fold][model][metric][1])
                else:
                    table += " & na" * len(metrics)
            table += "\\\\\n"
        table += "\hline\n"
    table += "\end{tabular}\n"
    table += "\end{table*}"
    print(table)





# def run_experiment_with_seed(seed, evaluation_modes, training_folds, models, exp_name):
#     """
#     Run a full experiment with a given seed.
#     :param seed: seed for reproducibility of experiments
#     :param evaluation_modes: list of evaluation modes for which the experiment has to be performed ("ml",
#     "ml\mr", "ml&mr")
#     :param training_folds: list of training datasets for which the experiment has to be performed ("ml",
#     "ml|mr(movies)", "ml|mr(movies+genres)", "ml(movies)|mr(genres)")
#     :param models: list of model on which the experiment has to be performed ("standard_mf", "ltn_mf", "ltn_mf_genres")
#     :param exp_name: name of experiment
#     """
#     # check if specified training folds are correct
#     assert all([fold in ["ml", "ml|mr(movies)", "ml|mr(movies+genres)", "ml(movies)|mr(genres)"]
#                 for fold in training_folds]), "One of the specified training folds is wrong."
#     # check if specified evaluation procedures are correct
#     assert all([mode in ["ml", "ml\mr", "ml&mr"] for mode in evaluation_modes]), "One of the specified evaluation " \
#                                                                                  "procedures is wrong."
#     assert all([model in ["standard_mf", "ltn_mf", 'ltn_mf_genres'] for model in models]), "One of the specified " \
#                                                                                            "models is wrong."
#     # create dataset manager
#     data_manager = DataManager("./datasets")
#     if "ml|mr(movies)" in training_folds:
#         # get dataset that is the union of ml and mr
#         movie_ratings, movie_genres_matrix, genre_ratings, idx_mapping = data_manager.get_ml_union_mr_dataset()
#     if "ml|mr(movies+genres)" in training_folds or "ml(movies)|mr(genres)" in training_folds:
#         # get dataset that is the union of movie ratings of ml and genre ratings of mr
#         g_movie_ratings, g_movie_genres_matrix, g_genre_ratings = data_manager.get_ml_union_mr_genre_ratings_dataset()
#     # for each evaluation mode, run the experiment
#     for mode in evaluation_modes:
#         # prepare datasets for given mode
#         datasets = []
#         # get ml-100k folds
#         ml = data_manager.get_ml100k_folds(seed, mode, n_neg=200)
#         if "ml" in training_folds:
#             datasets.append(ml)
#         if "ml|mr(movies)" in training_folds:
#             datasets.append(data_manager.get_fusion_folds_given_ml_folds(movie_ratings, ml.val, ml.test, idx_mapping))
#         if "ml|mr(movies+genres)" in training_folds:
#             data = data_manager.get_fusion_folds_given_ml_folds(movie_ratings, ml.val, ml.test, idx_mapping,
#                                                                 genre_ratings=genre_ratings)
#             data.set_item_genres_matrix(movie_genres_matrix)
#             datasets.append(data)
#         if "ml(movies)|mr(genres)" in training_folds:
#             data = data_manager.get_fusion_folds_given_ml_folds(g_movie_ratings, ml.val, ml.test,
#                                                                 genre_ratings=g_genre_ratings)
#             data.set_item_genres_matrix(g_movie_genres_matrix)
#             datasets.append(data)
#         # for each dataset, I have to perform the experiment
#         for dataset in datasets:
#             if "standard_mf" in models:
#                 standard_mf = StandardMFModel("%s-%s" % (mode, dataset.name))
#                 standard_mf.run_experiment(dataset, seed, exp_name)
#             if "ltn_mf" in models:
#                 ltn_mf = LTNMFModel("%s-%s" % (mode, dataset.name))
#                 ltn_mf.run_experiment(dataset, seed, exp_name)
#             if "ltn_mf_genres" in models:
#                 ltn_mf_genres = LTNMFGenresModel("%s-%s" % (mode, dataset.name))
#                 if isinstance(dataset, DatasetWithGenres):
#                     ltn_mf_genres.run_experiment(dataset, seed, exp_name)


# def train_and_test(trainer, tr_loader, val_loader, test_loader, test_metrics, result_file_name, experiment_name):
#     """
#     It trains and tests a model.
#     """
#     trainer.train(tr_loader, val_loader, "hit@10", n_epochs=100, early=10, verbose=1,
#                   save_path="./saved_models/%s.pth" % (experiment_name,))
#     trainer.load_model("./saved_models/%s.pth" % (experiment_name,))
#     result = trainer.test(test_loader, test_metrics)
#     append_to_result_file(result_file_name, experiment_name, result)
#
#
# def train_standard_mf(data, test_metrics, seed, experiment_name, result_file_name):
#     """
#     It trains and tests a standard MF model.
#     """
#     with open("./config/standard_mf.json") as json_file:
#         config = json.load(json_file)
#     set_seed(seed)
#     mf_model = MatrixFactorization(data.n_users, data.n_items, config['k'], config['biased'])
#     tr_loader = TrainingDataLoader(data.train, config['tr_batch_size'])
#     val_loader = ValDataLoader(data.val, config['val_batch_size'])
#     test_loader = ValDataLoader(data.test, config['val_batch_size'])
#     trainer = MFTrainer(mf_model, Adam(mf_model.parameters(), lr=config['lr'], weight_decay=config['wd']))
#     train_and_test(trainer, tr_loader, val_loader, test_loader, test_metrics, result_file_name, experiment_name)
#
#
# def train_ltn_mf(data, test_metrics, seed, experiment_name, result_file_name):
#     """
#     It trains and tests a MF model trained using an LTN.
#     """
#     with open("./config/ltn_mf.json") as json_file:
#         config = json.load(json_file)
#     set_seed(seed)
#     mf_model = MatrixFactorization(data.n_users, data.n_items, config['k'], config['biased'])
#     tr_loader = TrainingDataLoaderLTN(data.train, config['tr_batch_size'])
#     val_loader = ValDataLoader(data.val, config['val_batch_size'])
#     test_loader = ValDataLoader(data.test, config['val_batch_size'])
#     trainer = LTNTrainerMF(mf_model, Adam(mf_model.parameters(), lr=config['lr'], weight_decay=config['wd']),
#                            alpha=config['alpha'])
#     train_and_test(trainer, tr_loader, val_loader, test_loader, test_metrics, result_file_name, experiment_name)
#
#
# def train_ltn_mf_genres(data, test_metrics, seed, experiment_name, result_file_name):
#     """
#     It trains and tests a MF model trained using an LTN which adds also a formula to reason about the
#     genres of the movies.
#     """
#     with open("./config/ltn_mf_genres.json") as json_file:
#         config = json.load(json_file)
#     set_seed(seed)
#     mf_model = MatrixFactorization(data.n_users, data.n_items, config['k'], config['biased'])
#     # here, we pass n_items - n_genres because in this MF model the items include also the movie genres but
#     # the loader needs to know the number of movies (items without genres)
#     tr_loader = TrainingDataLoaderLTNGenres(data.train, data.n_users, data.n_items - data.n_genres, data.n_genres,
#                                             genre_sample_size=5, batch_size=config['tr_batch_size'])
#     val_loader = ValDataLoader(data.val, config['val_batch_size'])
#     test_loader = ValDataLoader(data.test, config['val_batch_size'])
#     # also the trainer needs to know the exact number of movies (n_items - n_genres)
#     trainer = LTNTrainerMFGenres(mf_model, Adam(mf_model.parameters(), lr=config['lr'], weight_decay=config['wd']),
#                                  alpha=config['alpha'], p=config['p'],
#                                  n_movies=data.n_items - data.n_genres, item_genres_matrix=data.item_genres_matrix)
#     train_and_test(trainer, tr_loader, val_loader, test_loader, test_metrics, result_file_name, experiment_name)
