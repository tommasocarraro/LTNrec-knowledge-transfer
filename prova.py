from ltnrec.data import DataManager
from ltnrec.models import StandardMFModel, LTNMFModel, LTNMFGenresModel
from ltnrec.data import DatasetWithGenres
from joblib import Parallel, delayed
from ltnrec.utils import create_report_dict, generate_report_table
import os
import pickle
import wandb


# login to Weights and Biases
wandb.login()
# create global wandb API object
api = wandb.Api()
api.entity = "bmxitalia"
# create tuples of available evaluation modes, datasets, and models in this repository
eval_modes = ("ml", "ml\mr", "ml&mr")
tr_folds = ("ml", "ml|mr(movies)", "ml|mr(movies+genres)", "ml(movies)|mr(genres)")
available_models = ("standard_mf", "ltn_mf", "ltn_mf_genres")
# create dataset manager
data_manager = DataManager("./datasets")
# get information for constructing the dataset that is the union of ml and mr
movie_ratings, movie_genres_matrix, genre_ratings, idx_mapping = data_manager.get_ml_union_mr_dataset()
# get information for constructing the dataset that is the union of movie ratings of ml and genre ratings of mr
g_movie_ratings, g_movie_genres_matrix, g_genre_ratings = data_manager.get_ml_union_mr_genre_ratings_dataset()


# define functions to perform serialization
# def grid_search(model, data, seed, save_path_prefix):
#     if not (isinstance(model, LTNMFGenresModel) and not isinstance(data, DatasetWithGenres)):
#         model.grid_search(data, seed, save_path_prefix)

def grid_search(model, dataset_id, local_dataset_path_prefix, local_config_path_prefix, wandb_project):
    # todo gestire il fatto che non posso lanciarla se il dataset non e' compatibile, sfruttare il nome del dataset
    model.grid_search_wandb(dataset_id, local_dataset_path_prefix, local_config_path_prefix, wandb_project)


def train(model, dataset_id, local_dataset_path_prefix, local_config_path_prefix, local_model_path_prefix,
          wandb_project):
    # if not (isinstance(model, LTNMFGenresModel) and not isinstance(data, DatasetWithGenres)):
    # todo gestire il fatto che non posso lanciarla se il dataset non e' compatibile, sfruttare il nome del dataset
    model.train_model_wandb(dataset_id, local_dataset_path_prefix, local_config_path_prefix, local_model_path_prefix,
                            wandb_project)


def model_test(model, dataset_id, local_dataset_path_prefix, local_config_path_prefix, local_model_path_prefix,
               local_result_path_prefix, wandb_project):
    # if not (isinstance(model, LTNMFGenresModel) and not isinstance(data, DatasetWithGenres)):
    # todo gestire il fatto che non posso lanciarla se il dataset non e' compatibile, sfruttare il nome del dataset
    model.test_model_wandb(dataset_id, local_dataset_path_prefix, local_config_path_prefix, local_model_path_prefix,
                           local_result_path_prefix, wandb_project)


def get_dataset_by_name(ml_dataset, name):
    """
    It constructs the requested dataset, specified by the `name` parameter, given the ml-100k dataset.

    :param ml_dataset: ml-100k dataset
    :param name: name of dataset that has to be generated
    :return: the requested dataset
    """
    if name == 'ml':
        return ml_dataset
    elif name == 'ml|mr(movies)':
        return data_manager.get_fusion_folds_given_ml_folds(movie_ratings, ml_dataset.val, ml_dataset.test, idx_mapping)
    elif name == 'ml|mr(movies+genres)':
        return data_manager.get_fusion_folds_given_ml_folds(movie_ratings, ml_dataset.val, ml_dataset.test, idx_mapping,
                                                            genre_ratings, movie_genres_matrix)
    elif name == 'ml(movies)|mr(genres)':
        return data_manager.get_fusion_folds_given_ml_folds(g_movie_ratings, ml_dataset.val, ml_dataset.test,
                                                            genre_ratings=g_genre_ratings,
                                                            item_genres_matrix=g_movie_genres_matrix)
    else:
        raise ValueError("The dataset with the given name cannot be generated. It is likely that the given "
                         "name is wrong. The available datasets are %s" % (", ".join(tr_folds), ))


# def create_dataset_artifact(run, artifact_name, dataset_local_path, artifact_file_name):
#     """
#     It creates a wandb dataset artifact.
#
#     :param run: wandb run that creates the artifact
#     :param artifact_name: name of artifact
#     :param dataset_local_path: local path of the file that has to be added to the wandb artifact
#     :param artifact_file_name: name of the file uploaded inside the artifact
#     :return:
#     """
#     dataset_artifact = wandb.Artifact(artifact_name, type="dataset")
#     dataset_artifact.add_file(dataset_local_path, name=artifact_file_name)
#     run.log_artifact(dataset_artifact)

def create_dataset_artifact(ml_dataset, dataset_name, artifact_name, dataset_local_path, wandb_project):
    with wandb.init(project=wandb_project, job_type="create_dataset") as run:
        run.name = "create_dataset:%s" % (artifact_name, )
        dataset_artifact = wandb.Artifact(artifact_name, type="dataset")
        # check if dataset already exists locally
        if not os.path.exists(dataset_local_path):
            # we create the dataset file if it does not exist
            dataset = get_dataset_by_name(ml_dataset, dataset_name)
            with open(dataset_local_path, 'wb') as dataset_file:
                # write dataset on disk
                print("Creating local file %s" % (dataset_local_path,))
                pickle.dump(dataset, dataset_file)
        dataset_artifact.add_file(dataset_local_path)
        # save the artifact on wandb
        print("Creating dataset artifact %s on project %s" % (artifact_name, wandb_project))
        run.log_artifact(dataset_artifact)


def create_datasets(mode, training_folds, seed, n_neg, wandb_project, local_path_prefix):
    """
    It creates the datasets specified by the parameter `training_folds`. It starts by creating the ml-100k dataset with
    the given mode, seed, and number of negative samples. Then, it constructs the requested datasets starting from
    the generated ml-100k.
    After the datasets are generated, they are uploaded as artifacts on wandb if an artifact of the dataset does not
    already exist.

    :param mode: evaluation mode which specifies how the ml-100k validation and test sets have to be created
    :param training_folds: name of datasets that have to be generated
    :param seed: random seed for reproducibility
    :param n_neg: number of negative items for each target item in validation/test sets
    :param wandb_project: name of the wandb project where the dataset artifacts have to be uploaded
    :param local_path_prefix: local path prefix where the created dataset files have to be stored
    """
    try:
        # check if artifact for ml-ml-n_neg with current seed already exists
        api.artifact("%s/%s-ml-%d-seed_%d:latest" % (wandb_project, mode, n_neg, seed))
    except wandb.errors.CommError:
        # if the exception is raised, it means that the dataset artifact has to be created
        ml_dataset = data_manager.get_ml100k_folds(seed, mode, n_neg=n_neg)
        create_dataset_artifact(ml_dataset, "ml", "%s-ml-%d-seed_%d" % (mode, n_neg, seed),
                                "%s/%s-ml-%d-seed_%d" % (local_path_prefix, mode, n_neg, seed), wandb_project)

    # create and upload all the requested datasets
    for dataset_name in training_folds:
        try:
            # check if dataset artifact already exists
            api.artifact("%s/%s-%s-%d-seed_%d:latest" % (wandb_project, mode, dataset_name, n_neg, seed))
            print("Dataset %s-%s-%d-seed_%d has been already created. Find the dataset "
                  "artifact at dataset/%s-%s-%d-seed_%d" % (mode, dataset_name, n_neg, seed, mode, dataset_name,
                                                            n_neg, seed))
        except wandb.errors.CommError:
            # if the exception is raised, it means that the dataset artifact has to be created
            # get ml-100k dataset artifact if the file is not saved locally, and create the file
            if not os.path.exists("%s/%s-ml-%d-seed_%d" % (local_path_prefix, mode, n_neg, seed)):
                with wandb.init(project=wandb_project, job_type="get_dataset") as run:
                    run.name = "get_dataset:%s-ml%d-seed_%d" % (mode, n_neg, seed)
                    run.use_artifact("%s-ml-%d-seed_%d:latest" % (mode, n_neg, seed)).download(local_path_prefix)

            # get ml-100k dataset
            with open("%s/%s-ml-%d-seed_%d" % (local_path_prefix, mode, n_neg, seed), 'rb') as dataset_file:
                ml_dataset = pickle.load(dataset_file)

            create_dataset_artifact(ml_dataset, dataset_name, "%s-%s-%d-seed_%d" % (mode, dataset_name, n_neg, seed),
                                    "%s/%s-%s-%d-seed_%d" % (local_path_prefix, mode, dataset_name, n_neg, seed),
                                    wandb_project)


def run_experiment(wandb_project, local_dataset_path_prefix, local_config_path_prefix, local_model_path_prefix,
                   local_result_path_prefix, evaluation_modes=eval_modes, training_folds=tr_folds, n_neg=200,
                   models=available_models, exp_name="experiment", starting_seed=0, n_runs=30,
                   num_workers=os.cpu_count()):
    """
    Run a full experiment with the given evaluation modes, training folds, and models. It runs the experiments for the
    given number of runs, starting with the given seed.
    It is possible to parallelize the grid searches and also the training of the best models.

    :param wandb_project: name of the WandB project that manages this experiment
    :param wandb_entity: name of the WandB entity that has to be used for this experiment (this is usually the
    username of the user on wandb)
    :param evaluation_modes: list of evaluation modes for which the experiment has to be performed ("ml",
    "ml\mr", "ml&mr")
    :param training_folds: list of training datasets for which the experiment has to be performed ("ml",
    "ml|mr(movies)", "ml|mr(movies+genres)", "ml(movies)|mr(genres)")
    :param n_neg: number of negative items that have to be sampled for each positive validation/test item
    :param models: list of model on which the experiment has to be performed ("standard_mf", "ltn_mf", "ltn_mf_genres")
    :param exp_name: name of experiment
    :param starting_seed: the seed of the first run of the experiment. The experiment is run for `n_runs` runs. The
    seed is progressive starting from `starting_seed`
    :param n_runs: number of runs of the experiment
    :param num_workers: Number of processors to use during the experiment. By default, it is the number of available
    processors in the computer. If num_workers is greater than 1, then the experiment is parallelized when it is
    possible
    """
    # verify that the number of runs is correct
    assert n_runs > 0, "You have specified a number of runs which is lower or equal to zero."
    # convert parameters to tuple if they are not tuples
    # todo fare una verifica perche' posso passare sia un solo elemento che una tupla, aggiungere questa verifica
    if not isinstance(evaluation_modes, tuple):
        evaluation_modes = (evaluation_modes, )
    if not isinstance(training_folds, tuple):
        training_folds = (training_folds, )
    if not isinstance(models, tuple):
        models = (models, )
    # check if specified training folds are correct
    assert all([fold in tr_folds for fold in training_folds]), "One of the specified training folds is wrong."
    # check if specified evaluation procedures are correct
    assert all([mode in eval_modes for mode in evaluation_modes]), "One of the specified evaluation " \
                                                                   "procedures is wrong."
    assert all([model in available_models for model in models]), "One of the specified models is wrong."

    # create folder to store datasets if it does not already exist
    if not os.path.exists(local_dataset_path_prefix):
        os.mkdir(local_dataset_path_prefix)

    # create folder to store configurations if it does not already exist
    if not os.path.exists(local_config_path_prefix):
        os.mkdir(local_config_path_prefix)

    # create folder to store models if it does not already exist
    if not os.path.exists(local_model_path_prefix):
        os.mkdir(local_model_path_prefix)

    # create folder to store results if it does not already exist
    if not os.path.exists(local_result_path_prefix):
        os.mkdir(local_result_path_prefix)

    # create list of models
    models_list = []
    if "standard_mf" in models:
        models_list.append(StandardMFModel())
    if "ltn_mf" in models:
        models_list.append(LTNMFModel())
    if "ltn_mf_genres" in models:
        models_list.append(LTNMFGenresModel())

    # set mode thread for wandb if the num of cpu is greater than 1
    if num_workers > 1:
        os.environ["WANDB_START_METHOD"] = "thread"

    # create datasets and save them
    Parallel(n_jobs=num_workers)(delayed(create_datasets)(mode, training_folds, seed, n_neg, wandb_project,
                                                          local_dataset_path_prefix)
                                 for mode in evaluation_modes
                                 for seed in range(starting_seed, starting_seed + n_runs))

    # run grid search of each model on each dataset with seed 0
    Parallel(n_jobs=num_workers)(delayed(grid_search)(model,
                                                      "%s-%s-%d-seed_%d" % (mode, dataset_name, n_neg, starting_seed),
                                                      local_dataset_path_prefix,
                                                      local_config_path_prefix,
                                                      wandb_project)
                                 for model in models_list
                                 for mode in evaluation_modes
                                 for dataset_name in training_folds)

    # train every model on every dataset with every seed with best configuration files obtained
    Parallel(n_jobs=num_workers)(delayed(train)(model,
                                                "%s-%s-%d-seed_%d" % (mode, dataset_name, n_neg, seed),
                                                local_dataset_path_prefix,
                                                local_config_path_prefix,
                                                local_model_path_prefix,
                                                wandb_project)
                                 for model in models_list
                                 for mode in evaluation_modes
                                 for dataset_name in training_folds
                                 for seed in range(starting_seed, starting_seed + n_runs))

    # test every model on every dataset with every seed with best model files obtained
    Parallel(n_jobs=num_workers)(delayed(model_test)(model,
                                                     "%s-%s-%d-seed_%d" % (mode, dataset_name, n_neg, seed),
                                                     local_dataset_path_prefix,
                                                     local_config_path_prefix,
                                                     local_model_path_prefix,
                                                     local_result_path_prefix,
                                                     wandb_project)
                                 for model in models_list
                                 for mode in evaluation_modes
                                 for dataset_name in training_folds
                                 for seed in range(starting_seed, starting_seed + n_runs))

    # todo sarebbe interessante fare una funzione make che genera loaders e modello
    # todo cercare il discorso di logging
    # todo fare una funzione utilita' che va a pescare tutti i result artifacts e mi crea un result artifact overall
    # todo capire come posso sovrascrivere una grid search per un dataset con un nuovo seed
    # todo mettere i messaggi di log come quelli per la creazione dei dataset in tutte le procedure
    # todo potrei salvare il miglior modello anche sulle grid search, solo il migliore ogni volta, cosi avviene
    #  versioning anche li
    # todo vorrei poter fare le grid search che voglio ma anche non doverla fare
    # todo capire bene cosa succede quando lancio la grid search direttamente dal training
    # todo idea: se esiste un best config artifact, allora seleziono quello per fare il training
    # wandb.login()
    # wandb.init(project='prova')
    #
    # artifact = wandb.Artifact('experiment-folds', type='dataset')
    # artifact.add_dir('./datasets/experiment-folds/seed_0/ml/', name="seed_0/ml")
    # wandb.log_artifact(artifact)

    # pool.starmap(create_dataset, [(mode, training_folds, datasets_dict, seed, wandb_artifacts)
    #                               for mode in evaluation_modes
    #                               for seed in range(starting_seed, starting_seed + n_runs)])
    # # get normal dict of datasets from serialized one
    # datasets_dict = {seed: {mode: dict(datasets_dict[seed])[mode] for mode in dict(datasets_dict[seed])}
    #                  for seed in dict(datasets_dict)}
    # pool.starmap(grid_search, [(model, dataset, starting_seed, "%s-%s" % (mode, dataset.name))
    #                            for model in models_list
    #                            for mode in datasets_dict[starting_seed]
    #                            for dataset in datasets_dict[starting_seed][mode]])
    # pool.starmap(train, [(model, dataset, seed, "%s-%s" % (mode, dataset.name), exp_name)
    #                      for seed in range(starting_seed, starting_seed + n_runs)
    #                      for model in models_list
    #                      for mode in datasets_dict[seed]
    #                      for dataset in datasets_dict[seed][mode]])


if __name__ == '__main__':
    run_experiment(wandb_project='prova',
                   local_dataset_path_prefix="./datasets/experiment-folds",
                   local_config_path_prefix="./config/best_config",
                   local_model_path_prefix="./saved_models/models",
                   local_result_path_prefix="./results/results",
                   evaluation_modes="ml",
                   training_folds="ml",
                   models="standard_mf",
                   starting_seed=0, n_runs=1)
    # todo verificare che tutto sia corretto e che effettivamente quelli siano i migliori iperparametri
    # todo applicare un metodo tipo optuna o wandb e verificare se ci sono parametri migliori
    # todo forse bisognerebbe usare il logger per fare in modo che le cose vengano loggate correttamente anche su wandb
    # todo vorrei una struttura un po' meglio per i file
    # todo non mi piacere vedere tutte le run di una grid search per ogni modello, vorrei vedere solo una grid search per modello
    # report_dict = create_report_dict(starting_seed=0, n_runs=30, exp_name="experiment")
    # pprint.pprint(report_dict)
    # generate_report_table(report_dict, metrics=("ndcg@10",))
    # pprint.pprint(report_dict)




# def create_dataset(mode, training_folds, datasets_dict, seed, wandb_artifacts=False, wandb_project=None):
#     # todo potrei avere un parametro wandb_artifacts che mi salva le cose se e' a true, ricordarsi del wandb.finish()
#     # todo funziona tutto, vedere il discorso dei tag
#     if not os.path.exists("./datasets/experiment-folds/seed_%d" % (seed, )):
#         print("Creating folder for datasets with seed %s" % (seed, ))
#         os.mkdir("./datasets/experiment-folds/seed_%d" % (seed, ))
#     if not os.path.exists("./datasets/experiment-folds/seed_%d/%s" % (seed, mode)):
#         print("Creating folder for datasets with evaluation mode %s and seed %d" % (mode, seed))
#         os.mkdir("./datasets/experiment-folds/seed_%d/%s" % (seed, mode))
#     print("Starting procedure of creation of datasets with evaluation mode %s and seed %d" % (mode, seed))
#     # prepare datasets for given mode and seed
#     datasets = []
#     if wandb_artifacts:
#         assert wandb_project is not None, "If you want to use wandb artifacts, you need to specify a wandb project name"
#         wandb_run = wandb.init(project=wandb_project, job_type="upload/load")
#     # get ml-100k folds
#     if os.path.exists("./datasets/experiment-folds/seed_%d/%s/ml.dataset" % (seed, mode)):
#         print("ml dataset for evaluation mode %s and seed %d already exists. Loading dataset from disk." % (mode, seed))
#         with open("./datasets/experiment-folds/seed_%d/%s/ml.dataset" % (seed, mode), 'rb') as dataset_file:
#             ml = pickle.load(dataset_file)
#     else:
#         # create MovieLens dataset with given mode
#         print("Creating ml dataset for evaluation mode %s and seed %d" % (mode, seed))
#         ml = data_manager.get_ml100k_folds(seed, mode, n_neg=200)
#         # save dataset to disk
#         with open("./datasets/experiment-folds/seed_%d/%s/ml.dataset" % (seed, mode), 'wb') as dataset_file:
#             pickle.dump(ml, dataset_file)
#             if wandb_artifacts:
#                 # create dataset artifact if requested by the user
#                 create_dataset_artifact(wandb_run, "%s-ml-seed_%d" % (mode, seed),
#                                         "./datasets/experiment-folds/seed_%d/%s/ml.dataset" % (seed, mode), "dataset")
#
#     for dataset_name in training_folds:
#         if os.path.exists("./datasets/experiment-folds/seed_%d/%s/%s.dataset" % (seed, mode, dataset_name)):
#             print("%s dataset for evaluation mode %s and seed %d already exists. Loading dataset from disk." % (
#                   dataset_name, mode, seed))
#             with open("./datasets/experiment-folds/seed_%d/%s/%s.dataset" % (seed, mode, dataset_name), 'rb') as dataset_file:
#                 dataset = pickle.load(dataset_file)
#         else:
#             print("Creating %s dataset for evaluation mode %s and seed %d" % (dataset_name, mode, seed))
#             dataset = get_dataset_by_name(ml, dataset_name)
#             # save dataset to disk
#             print("Saving %s dataset for evaluation mode %s and seed %d" % (dataset_name, mode, seed))
#             with open("./datasets/experiment-folds/seed_%d/%s/%s.dataset" % (seed, mode, dataset_name), 'wb') as dataset_file:
#                 pickle.dump(dataset, dataset_file)
#                 if wandb_artifacts:
#                     # create dataset artifact if requested by the user
#                     dataset_artifact = wandb.Artifact("%s-ml-seed_%d" % (mode, seed), type="dataset")
#                     dataset_artifact.add_file("./datasets/experiment-folds/seed_%d/%s/ml.dataset" % (seed, mode),
#                                               name="dataset")
#                     wandb_run.log_artifact(dataset_artifact)
#
#         datasets.append(dataset)
#
#     datasets_dict[seed][mode] = datasets

