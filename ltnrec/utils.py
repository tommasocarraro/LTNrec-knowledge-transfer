import torch
import numpy as np
import random
import json
import os


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


def remove_seed_from_dataset_name(dataset_name):
    """
    It remove the seed from the dataset name.

    :param dataset_name: name of the dataset
    :return:
    """
    return "_".join(dataset_name.split("_")[:-1]) + "_"


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


def create_report_json(local_result_path_prefix, save_path, seeds, json_file_name="experiment-results"):
    # create dictionary containing the results
    result_dict = {}
    # iterate over created result files
    for result_file in os.listdir(local_result_path_prefix):
        # check that the file is a JSON file and begins with 'result-' and the seed is one of the requested
        if result_file.endswith(".json") and result_file.startswith("result-") and int(result_file.split(".json")[0].split("_")[-1]) in seeds:
            # get file content
            with open(os.path.join(local_result_path_prefix, result_file), "r") as json_file:
                metrics_values = json.load(json_file)
            # get model of the experiment
            model = result_file.split("-")[1].split("=")[1]
            # get evaluation mode of the experiment
            evaluation_mode = result_file.split("-")[2].split("=")[1]
            # get training fold of the experiment
            training_fold = result_file.split("-")[3]
            # get proportion of training ratings per user
            prop = result_file.split("-")[5]

            # put results at the right place in the big report dict
            if evaluation_mode not in result_dict:
                result_dict[evaluation_mode] = {training_fold: {prop: {model: {metric: [metrics_values[metric]]
                                                                               for metric in metrics_values}}}}
            elif training_fold not in result_dict[evaluation_mode]:
                result_dict[evaluation_mode][training_fold] = {prop: {model: {metric: [metrics_values[metric]]
                                                                              for metric in metrics_values}}}
            elif prop not in result_dict[evaluation_mode][training_fold]:
                result_dict[evaluation_mode][training_fold][prop] = {model: {metric: [metrics_values[metric]]
                                                                             for metric in metrics_values}}
            elif model not in result_dict[evaluation_mode][training_fold][prop]:
                result_dict[evaluation_mode][training_fold][prop][model] = {metric: [metrics_values[metric]]
                                                                            for metric in metrics_values}
            else:
                for metric in metrics_values:
                    result_dict[evaluation_mode][training_fold][prop][model][metric].append(metrics_values[metric])

    # loop over the big report dict to compute mean and std of the experiments with different seeds
    for evaluation_mode in result_dict:
        for training_fold in result_dict[evaluation_mode]:
            for prop in result_dict[evaluation_mode][training_fold]:
                for model in result_dict[evaluation_mode][training_fold][prop]:
                    for metric in result_dict[evaluation_mode][training_fold][prop][model]:
                        result_dict[evaluation_mode][training_fold][prop][model][metric] = (
                            np.mean(result_dict[evaluation_mode][training_fold][prop][model][metric]),
                            np.std(result_dict[evaluation_mode][training_fold][prop][model][metric]))

    # create JSON file with the result
    with open(os.path.join(save_path, json_file_name + ".json"), "w") as outfile:
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
                          training_folds=("ml", "ml|mr(movies)", "ml|mr(movies+genres)", "ml(movies)|mr(genres)"),
                          proportions_to_keep=(1, )):
    # check if report_dict is a dict - if it is not a dict, it has to be a path to a JSON file
    if not isinstance(report_dict, dict):
        with open(report_dict, "r") as json_file:
            report_dict = json.load(json_file)

    table = "\\begin{table*}[ht!]\n"
    table += "\caption{Table title}\label{table-label}\n"
    table += "\centering\n"
    table += "\\begin{tabular}{| c | c | %s %s}\n" % ("c | " if len(proportions_to_keep) != 1 else "",
                                                      "".join(["c " * len(metrics) + "| " for _ in models], ))
    table += "\hline\n"
    table += "&"
    if len(proportions_to_keep) != 1:
        table += " &"
    for model in models:
        table += " & \multicolumn{%d}{c |}{%s}" % (len(metrics), model.replace("_", "\\_"))
    table += "\\\\\n"
    table += "\hline\n"
    table += "&"
    if len(proportions_to_keep) != 1:
        table += " & \\% ratings"
    for _ in models:
        for metric in metrics:
            table += " & %s" % (metric, )
    table += "\\\\\n"
    table += "\hline\n"
    for mode in evaluation_modes:
        table += "\multirow{%d}{*}{%s}" % (len(training_folds) * len(proportions_to_keep), mode.replace("\\", " $\setminus$ ").replace("&", " $\cap$ "))
        for tr_fold in training_folds:
            if len(proportions_to_keep) != 1:
                table += " & \multirow{%d}{*}{%s}" % (len(proportions_to_keep), tr_fold.replace("|", " $\cup$ "), )
                for idx, prop in enumerate(proportions_to_keep):
                    if idx != 0:
                        table += " &"
                    table += " & %d\\%s" % (prop * 100, "%")
                    for model in models:
                        prop_str = "%.2f" % (prop, )
                        if tr_fold in report_dict[mode] and prop_str in report_dict[mode][tr_fold] and model in report_dict[mode][tr_fold][prop_str]:
                            for metric in metrics:
                                table += " & %.3f$_{(%.3f)}$" % (report_dict[mode][tr_fold][prop_str][model][metric][0],
                                                                 report_dict[mode][tr_fold][prop_str][model][metric][1])
                        else:
                            table += " & na" * len(metrics)
                    table += "\\\\\n"
            else:
                table += " & %s" % (tr_fold.replace("|", " $\cup$ "), )
                for model in models:
                    prop_str = "%.2f" % (1, )
                    if tr_fold in report_dict[mode] and model in report_dict[mode][tr_fold][prop_str]:
                        for metric in metrics:
                            table += " & %.3f$_{(%.3f)}$" % (report_dict[mode][tr_fold][prop_str][model][metric][0],
                                                             report_dict[mode][tr_fold][prop_str][model][metric][1])
                    else:
                        table += " & na" * len(metrics)
                table += "\\\\\n"
            if len(proportions_to_keep) != 1:
                table += "\cline{2-%d}\n" % (len(metrics) * len(models) + 3)

        table += "\hline\n"
    table += "\end{tabular}\n"
    table += "\end{table*}"
    print(table)
