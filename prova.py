from ltnrec.data import DataManager
from ltnrec.models import StandardMFModel, LTNMFModel, LTNMFGenresModel
from ltnrec.data import DatasetWithGenres
from multiprocessing import Manager
from joblib import Parallel, delayed
from ltnrec.utils import create_report_dict, generate_report_table
import os
import pickle
import wandb

eval_modes = ("ml", "ml\mr", "ml&mr")
tr_folds = ("ml", "ml|mr(movies)", "ml|mr(movies+genres)", "ml(movies)|mr(genres)")
available_models = ("standard_mf", "ltn_mf", "ltn_mf_genres")
# create dataset manager
data_manager = DataManager("./datasets")
# get dataset that is the union of ml and mr
movie_ratings, movie_genres_matrix, genre_ratings, idx_mapping = data_manager.get_ml_union_mr_dataset()
# get dataset that is the union of movie ratings of ml and genre ratings of mr
g_movie_ratings, g_movie_genres_matrix, g_genre_ratings = data_manager.get_ml_union_mr_genre_ratings_dataset()


# define functions to perform serialization
# def grid_search(model, data, seed, save_path_prefix):
#     # todo carica il dataset da disco o da wandb e poi chiama la grid search corretta con quel dataset
#     if not (isinstance(model, LTNMFGenresModel) and not isinstance(data, DatasetWithGenres)):
#         model.grid_search(data, seed, save_path_prefix)

def grid_search(model, dataset_id, wandb_artifacts=False, wandb_project=None, wandb_entity=None):
    if wandb_artifacts:
        # load dataset from wandb artifact
        wandb_run = wandb.init(project=wandb_project, job_type="grid_search")
        wandb_run.name = "grid_search-model_%s-on-%s" % (model.model_name, dataset_id)
        wandb_run.use_artifact("%s/%s/%s:latest" % (wandb_entity, wandb_project, dataset_id)).download(
            "./datasets/experiment-folds/")
    else:
        # load dataset from disk
        wandb_run = None

    print("Loading dataset %s" % (dataset_id, ))
    with open("./datasets/experiment-folds/%s" % (dataset_id, ), 'rb') as dataset_file:
        data = pickle.load(dataset_file)

    if not (isinstance(model, LTNMFGenresModel) and not isinstance(data, DatasetWithGenres)):
        print("Starting grid search of model %s on dataset %s" % (model.model_name, dataset_id))
        model.grid_search(data, seed)


def train(model, data, seed, save_path_prefix, exp_name):
    if not (isinstance(model, LTNMFGenresModel) and not isinstance(data, DatasetWithGenres)):
        model.run_experiment(data, seed, save_path_prefix, exp_name)


def get_dataset_by_name(ml_dataset, name):
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
        raise ValueError("The dataset with the given name does not exist.")


def create_dataset_artifact(run, artifact_name, dataset_local_path, artifact_file_name):
    dataset_artifact = wandb.Artifact(artifact_name, type="dataset")
    dataset_artifact.add_file(dataset_local_path, name=artifact_file_name)
    run.log_artifact(dataset_artifact)


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


def create_dataset(mode, training_folds, seed, n_neg, wandb_project=None, wandb_entity=None):
    created_something = None
    # create datasets with given mode and seed
    print("Starting procedure of creation of datasets with evaluation mode %s and seed %d" % (mode, seed))
    if wandb_artifacts:
        assert wandb_project is not None, "If you want to use wandb artifacts, you need to specify a wandb project name"
        # create a wandb run that upload the datasets as artifacts
        wandb_run = wandb.init(project=wandb_project, job_type="upload_dataset")
        wandb_run.name = "create-datasets-evaluation_mode_%s-seed_%d" % (mode, seed)
        created_something = False
    else:
        wandb_run = None
    # get ml-100k folds
    if os.path.exists("./datasets/experiment-folds/%s-ml-seed_%d" % (mode, seed)):
        print("ml dataset for evaluation mode %s and seed %d already exists." % (mode, seed))
        with open("./datasets/experiment-folds/%s-ml-seed_%d" % (mode, seed), 'rb') as dataset_file:
            # take dataset since it is used to create the other datasets
            ml = pickle.load(dataset_file)
    else:
        # create MovieLens dataset with given mode
        print("Creating ml dataset for evaluation mode %s and seed %d" % (mode, seed))
        ml = data_manager.get_ml100k_folds(seed, mode, n_neg=n_neg)
        # save dataset to disk
        with open("./datasets/experiment-folds/%s-ml-seed_%d" % (mode, seed), 'wb') as dataset_file:
            pickle.dump(ml, dataset_file)
        if wandb_artifacts:
            # create dataset artifact if requested by the user
            print("Creating artifact for ml dataset for evaluation mode %s and seed %d" % (mode, seed))
            create_dataset_artifact(wandb_run, "%s-ml-seed_%d" % (mode, seed),
                                    "./datasets/experiment-folds/%s-ml-seed_%d" % (mode, seed),
                                    "%s-ml-seed_%d" % (mode, seed))
            created_something = True

    for dataset_name in training_folds:
        if os.path.exists("./datasets/experiment-folds/%s-%s-seed_%d" % (mode, dataset_name, seed)):
            print("%s dataset for evaluation mode %s and seed %d already exists." % (dataset_name, mode, seed))
        else:
            print("Creating %s dataset for evaluation mode %s and seed %d" % (dataset_name, mode, seed))
            dataset = get_dataset_by_name(ml, dataset_name)
            # save dataset to disk
            print("Saving %s dataset for evaluation mode %s and seed %d" % (dataset_name, mode, seed))
            with open("./datasets/experiment-folds/%s-%s-seed_%d" % (mode, dataset_name, seed), 'wb') as dataset_file:
                pickle.dump(dataset, dataset_file)
            if wandb_artifacts:
                # create dataset artifact if requested by the user
                print("Creating artifact for %s dataset for evaluation mode %s and seed %d" % (dataset_name, mode,
                                                                                               seed))
                create_dataset_artifact(wandb_run, "%s-%s-seed_%d" % (mode, dataset_name, seed),
                                        "./datasets/experiment-folds/%s-%s-seed_%d" % (mode, dataset_name, seed),
                                        "%s-%s-seed_%d" % (mode, dataset_name, seed))
                created_something = True

    # end wandb run
    if wandb_artifacts:
        wandb_run.finish()
        if not created_something:
            # delete run if it has not created anything since it is not useful to display it
            api = wandb.Api()
            run = api.run("%s/%s/%s" % (wandb_entity, wandb_project, wandb_run.id))
            run.delete()


def run_experiment(evaluation_modes=eval_modes, training_folds=tr_folds, n_neg=200, models=available_models,
                   exp_name="experiment", starting_seed=0, n_runs=30, num_workers=os.cpu_count(),
                   wandb_project=None, wandb_entity=None, wandb_artifacts=False):
    """
    Run a full experiment with the given evaluation modes, training folds, and models. It runs the experiments for the
    given number of runs, starting with the given seed.
    It is possible to parallelize the grid searches and also the training of the best models.

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
    :param wandb_project: name of the WandB project that manages this experiment
    :param wandb_entity: name of the WandB entity that has to be used for these experiments
    :param wandb_artifacts: whether WandB artifacts have to be created to store the datasets and models during the
    experiment
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
    # create list of models
    models_list = []
    if "standard_mf" in models:
        models_list.append(StandardMFModel())
    if "ltn_mf" in models:
        models_list.append(LTNMFModel())
    if "ltn_mf_genres" in models:
        models_list.append(LTNMFGenresModel())

    # create dictionary of datasets for different seeds - this can be done in parallel
    manager = Manager()
    datasets_dict = manager.dict({seed: manager.dict({mode: [] for mode in evaluation_modes})
                                 for seed in range(starting_seed, starting_seed + n_runs)})
    if not os.path.exists("./datasets/experiment-folds"):
        print("Creating folder for datasets that have to be created by the experiment")
        os.mkdir("./datasets/experiment-folds")

    if wandb_project is not None:
        wandb.login()
        if num_workers > 1:
            os.environ["WANDB_START_METHOD"] = "thread"

    # create datasets and save them
    Parallel(n_jobs=num_workers)(delayed(create_dataset)(mode, training_folds, seed, n_neg, wandb_artifacts,
                                                         wandb_project, wandb_entity)
                                 for mode in evaluation_modes
                                 for seed in range(starting_seed, starting_seed + n_runs))

    # run grid search of each model on each dataset with seed 0
    Parallel(n_jobs=num_workers)(delayed(grid_search)(model, "%s-%s-seed_%d" % (mode, dataset_name, starting_seed),
                                                      True, wandb_project, wandb_entity)
                                 for model in models_list
                                 for mode in evaluation_modes
                                 for dataset_name in training_folds)


    # todo posso semplicemente passare il nome dell'artifact alla procedura di grid search, in maniera tale che mi carica
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
    run_experiment(starting_seed=0, n_runs=2, wandb_artifacts=True, wandb_project='prova', wandb_entity='bmxitalia')
    # todo verificare che tutto sia corretto e che effettivamente quelli siano i migliori iperparametri
    # todo applicare un metodo tipo optuna o wandb e verificare se ci sono parametri migliori
    # report_dict = create_report_dict(starting_seed=0, n_runs=30, exp_name="experiment")
    # pprint.pprint(report_dict)
    # generate_report_table(report_dict, metrics=("ndcg@10",))
    # pprint.pprint(report_dict)

