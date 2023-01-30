from ltnrec.data import DataManager
from ltnrec.models import StandardMFModel, LTNMFModel, LTNMFGenresModel
from joblib import Parallel, delayed
from ltnrec.utils import create_report_json, generate_report_table
import os
import pickle
import wandb


# login to Weights and Biases
wandb.login()
# create global wandb API object
api = wandb.Api()
api.entity = "bmxitalia"
# create tuples of available evaluation modes, datasets, and models in this repository
eval_modes = ("ml", "ml&mr", "ml\mr")
tr_folds = ("ml", "ml|mr(movies)", "ml|mr(movies+genres)", "ml(movies)|mr(genres)")
available_models = ("standard_mf", "ltn_mf", "ltn_mf_genres")
tr_folds_mr = ("mr(movies)", "mr(movies+genres)")
# create dataset manager
data_manager = DataManager("./datasets")
# get information for constructing the dataset that is the union of ml and mr
movie_ratings, movie_genres_matrix, genre_ratings, idx_mapping = data_manager.get_ml_union_mr_dataset()
# get information for constructing the dataset that is the union of movie ratings of ml and genre ratings of mr
g_movie_ratings, g_movie_genres_matrix, g_genre_ratings = data_manager.get_ml_union_mr_genre_ratings_dataset()


def check_compatibility(model, dataset_id):
    return not (isinstance(model, LTNMFGenresModel) and "genres" not in dataset_id)


def grid_search(model, dataset_id, local_dataset_path_prefix, local_config_path_prefix, wandb_project):
    model.grid_search_wandb(dataset_id, local_dataset_path_prefix, local_config_path_prefix, wandb_project)


def train(model, dataset_id, config_seed, local_dataset_path_prefix, local_config_path_prefix, local_model_path_prefix,
          wandb_project):
    model.train_model_wandb(dataset_id, config_seed, local_dataset_path_prefix, local_config_path_prefix,
                            local_model_path_prefix, wandb_project)


def model_test(model, dataset_id, config_seed, local_dataset_path_prefix, local_config_path_prefix,
               local_model_path_prefix, local_result_path_prefix, wandb_project):
    model.test_model_wandb(dataset_id, config_seed, local_dataset_path_prefix, local_config_path_prefix,
                           local_model_path_prefix, local_result_path_prefix, wandb_project)


def get_dataset_by_name(ml_dataset, training_fold, p_to_keep=1, seed=None):
    """
    It constructs the requested dataset, specified by the `name` parameter, given the ml-100k dataset.
    It controls the sparsity of the generated dataset according to parameter `p_to_keep`.

    :param ml_dataset: ml-100k dataset
    :param training_fold: name of training fold that has to be generated
    :param p_to_keep: proportion of ratings to be kept in the training set. Default to 1. It means that all the ratings
    are kept, hence the dataset maintains the original sparsity
    :param seed: seed for random sampling in the case `p_to_keep` is different from 1.
    :return: the requested dataset
    """
    if training_fold == 'ml':
        return data_manager.increase_data_sparsity(ml_dataset, p_to_keep, seed)
    elif training_fold == 'ml|mr(movies)':
        return data_manager.increase_data_sparsity(data_manager.get_fusion_folds_given_ml_folds(
            movie_ratings, ml_dataset.val, ml_dataset.test, idx_mapping), p_to_keep, seed)
    elif training_fold == 'ml|mr(movies+genres)':
        return data_manager.increase_data_sparsity(data_manager.get_fusion_folds_given_ml_folds(
            movie_ratings, ml_dataset.val, ml_dataset.test, idx_mapping, genre_ratings, movie_genres_matrix),
            p_to_keep, seed)
    elif training_fold == 'ml(movies)|mr(genres)':
        return data_manager.increase_data_sparsity(data_manager.get_fusion_folds_given_ml_folds(
            g_movie_ratings, ml_dataset.val, ml_dataset.test, genre_ratings=g_genre_ratings,
            item_genres_matrix=g_movie_genres_matrix), p_to_keep, seed)
    else:
        raise ValueError("The dataset with the given name cannot be generated. It is likely that the given "
                         "name is wrong. The available datasets are %s" % (", ".join(tr_folds), ))


def create_dataset(local_dataset_path_prefix, ml_dataset, training_fold, dataset_name, p_to_keep=1, seed=None):
    """
    Creates a dataset with the given information.

    :param local_dataset_path_prefix: local path where to save the dataset
    :param ml_dataset: ml-100k dataset that has to be used to create the required dataset. In particular, it is used
    to create the same validation and test folds in the new dataset, in order to have a correct comparison.
    :param training_fold: name of the dataset that has to be created
    :param dataset_name: name of the dataset to display it
    :param p_to_keep: number of ratings that have to be kept for each user in the training set of the dataset
    :param seed: seed for random sampling of ratings that have to be kept
    """
    print("Creating dataset %s" % (dataset_name, ))
    # generate requested dataset from ml-100k
    dataset = get_dataset_by_name(ml_dataset, training_fold, p_to_keep, seed)
    # create dataset pickle file
    with open(os.path.join(local_dataset_path_prefix, dataset_name), 'wb') as dataset_file:
        pickle.dump(dataset, dataset_file)


def create_datasets(mode, training_folds, seed, n_neg, local_dataset_path_prefix,
                    proportions_to_keep=(1, )):
    """
    It creates the datasets specified by the parameter `training_folds` with the proportions of ratings per user
    specified in `proportions_to_keep`. It starts by creating the ml-100k dataset with
    the given mode, seed, number of negative samples, and proportion of ratings to keep equal to 1 (entire
    ml-100k dataset).
    Then, it constructs the requested datasets starting from the generated ml-100k.

    :param mode: evaluation mode which specifies how the ml-100k validation and test sets have to be created
    :param training_folds: name of datasets that have to be generated
    :param seed: random seed for reproducibility
    :param n_neg: number of negative items for each target item in validation/test sets
    :param local_dataset_path_prefix: local path prefix where the created dataset files have to be stored
    :param proportions_to_keep: tuple of proportions (between 0 and 1) of ratings that have to be kept for each user
    in the training sets of the datasets. It is used to change the sparsity of the dataset. Default to (1, ), indicating
    that all the ratings have to be kept.
    """
    ml_dataset = None
    # check if ml-ml-n_neg dataset with current seed and all the ratings already exists
    if not os.path.exists(os.path.join(local_dataset_path_prefix, "%s-ml-%d-%.2f-seed_%d" % (mode, n_neg, 1, seed))):
        # if the dataset does not exist, we need to create it
        ml_dataset = data_manager.get_ml100k_folds(seed, mode, n_neg=n_neg)
        create_dataset(local_dataset_path_prefix, ml_dataset, "ml", "%s-ml-%d-%.2f-seed_%d" % (mode, n_neg, 1, seed))

    if ml_dataset is None:
        # load ml-100k dataset
        with open("%s/%s-ml-%d-%.2f-seed_%d" % (local_dataset_path_prefix, mode, n_neg, 1, seed), 'rb') as dataset_file:
            ml_dataset = pickle.load(dataset_file)

    # create and upload all the requested datasets
    for dataset_name in training_folds:
        for p in proportions_to_keep:
            # check if dataset already exists
            if not os.path.exists(os.path.join(local_dataset_path_prefix, "%s-%s-%d-%.2f-seed_%d" % (mode, dataset_name,
                                                                                                     n_neg, p, seed))):
                # create the dataset
                create_dataset(local_dataset_path_prefix, ml_dataset, dataset_name, "%s-%s-%d-%.2f-seed_%d" %
                               (mode, dataset_name, n_neg, p, seed), p, seed)


def create_datasets_mr(training_folds, seed, n_neg, local_dataset_path_prefix, proportions_to_keep=(1, )):
    # create and upload all the requested datasets
    for dataset_name in training_folds:
        for p in proportions_to_keep:
            # check if dataset already exists
            if not os.path.exists(os.path.join(local_dataset_path_prefix, "mr-%s-%d-%.2f-seed_%d" % (dataset_name,
                                                                                                     n_neg, p, seed))):
                print("Creating dataset mr-%s-%d-%.2f-seed_%d" % (dataset_name, n_neg, p, seed))
                # generate requested dataset
                if dataset_name == "mr(movies)":
                    dataset = data_manager.increase_data_sparsity(data_manager.get_mr_folds(seed, n_neg), p_to_keep=p,
                                                                  seed=seed)
                elif dataset_name == "mr(movies+genres)":
                    dataset = data_manager.increase_data_sparsity(data_manager.get_mr_folds(seed, n_neg,
                                                                                            with_genres=True),
                                                                  p_to_keep=p, seed=seed)
                else:
                    raise ValueError("One of the given training fold names is not correct. Available training folds "
                                     "for MindReader experiments are %s" % (", ".join(tr_folds_mr), ))
                # create dataset pickle file
                with open(os.path.join(local_dataset_path_prefix,
                                       "mr-%s-%d-%.2f-seed_%d" % (dataset_name, n_neg, p, seed)), 'wb') as dataset_file:
                    pickle.dump(dataset, dataset_file)


def run_experiment(wandb_project, evaluation_modes=eval_modes, training_folds=tr_folds, n_neg=100,
                   models=available_models, proportions_to_keep=(1, ), starting_seed=0, n_runs=30,
                   num_workers=os.cpu_count()):
    """
    Run a full experiment with the given evaluation modes, training folds, and models. It runs the experiments for the
    given number of runs, starting with the given seed.
    It is possible to parallelize the grid searches and also the training of the best models.

    :param wandb_project: name of the WandB project that manages this experiment
    :param evaluation_modes: list of evaluation modes for which the experiment has to be performed ("ml",
    "ml\mr", "ml&mr")
    :param training_folds: list of training datasets for which the experiment has to be performed ("ml",
    "ml|mr(movies)", "ml|mr(movies+genres)", "ml(movies)|mr(genres)")
    :param n_neg: number of negative items that have to be sampled for each positive validation/test item
    :param models: list of model on which the experiment has to be performed ("standard_mf", "ltn_mf", "ltn_mf_genres")
    :param proportions_to_keep: different sparsity levels of the datasets that have to be created. For each
    evaluation_mode-tr_fold pair, it creates the datasets with the given sparsity levels.
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
    assert all([model in available_models for model in models]), "One of the specified models is wrong. You gave" \
                                                                 "the following tuple of " \
                                                                 "models: %s" % (" ".join(models), )

    local_dataset_path_prefix = "./%s/datasets" % (wandb_project, )
    local_config_path_prefix = "./%s/configs" % (wandb_project, )
    local_model_path_prefix = "./%s/models" % (wandb_project, )
    local_result_path_prefix = "./%s/results" % (wandb_project, )

    # create folder to store datasets if it does not already exist
    if not os.path.exists(local_dataset_path_prefix):
        os.makedirs(local_dataset_path_prefix)

    # create folder to store configurations if it does not already exist
    if not os.path.exists(local_config_path_prefix):
        os.makedirs(local_config_path_prefix)

    # create folder to store models if it does not already exist
    if not os.path.exists(local_model_path_prefix):
        os.makedirs(local_model_path_prefix)

    # create folder to store results if it does not already exist
    if not os.path.exists(local_result_path_prefix):
        os.makedirs(local_result_path_prefix)

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
    Parallel(n_jobs=num_workers)(delayed(create_datasets)(mode, training_folds, seed, n_neg,
                                                          local_dataset_path_prefix, proportions_to_keep)
                                 for mode in evaluation_modes
                                 for seed in range(starting_seed, starting_seed + n_runs))

    # run grid search of each model on each dataset with seed 0 and proportion 1
    Parallel(n_jobs=num_workers)(delayed(grid_search)(model,
                                                      "%s-%s-%d-%.2f-seed_%d" % (mode, dataset_name, n_neg, p, starting_seed),
                                                      local_dataset_path_prefix,
                                                      local_config_path_prefix,
                                                      wandb_project)
                                 for model in models_list
                                 for mode in evaluation_modes
                                 for dataset_name in training_folds
                                 for p in proportions_to_keep
                                 if check_compatibility(model, "%s-%s-%d-seed_%d" % (mode, dataset_name, n_neg,
                                                                                     starting_seed)))

    # # train every model on every dataset with every seed with best configuration files obtained
    Parallel(n_jobs=num_workers)(delayed(train)(model,
                                                "%s-%s-%d-%.2f-seed_%d" % (mode, dataset_name, n_neg, p, seed),
                                                starting_seed,
                                                local_dataset_path_prefix,
                                                local_config_path_prefix,
                                                local_model_path_prefix,
                                                wandb_project)
                                 for model in models_list
                                 for mode in evaluation_modes
                                 for dataset_name in training_folds
                                 for seed in range(starting_seed, starting_seed + n_runs)
                                 for p in proportions_to_keep
                                 if check_compatibility(model, "%s-%s-%d-seed_%d" % (mode, dataset_name, n_neg,
                                                                                     starting_seed)))
    # # #
    # # # # test every model on every dataset with every seed with best model files obtained
    Parallel(n_jobs=num_workers)(delayed(model_test)(model,
                                                     "%s-%s-%d-%.2f-seed_%d" % (mode, dataset_name, n_neg, p, seed),
                                                     starting_seed,
                                                     local_dataset_path_prefix,
                                                     local_config_path_prefix,
                                                     local_model_path_prefix,
                                                     local_result_path_prefix,
                                                     wandb_project)
                                 for model in models_list
                                 for mode in evaluation_modes
                                 for dataset_name in training_folds
                                 for seed in range(starting_seed, starting_seed + n_runs)
                                 for p in proportions_to_keep
                                 if check_compatibility(model, "%s-%s-%d-seed_%d" % (mode, dataset_name, n_neg,
                                                                                     starting_seed)))
    #
    create_report_json(local_result_path_prefix, "./%s" % (wandb_project, ), seeds=list(range(starting_seed, n_runs)))
    generate_report_table("./%s/experiment-results.json" % (wandb_project, ),
                          models=models, evaluation_modes=evaluation_modes, training_folds=training_folds)

    # todo sarebbe interessante fare una funzione make che genera loaders e modello
    # todo capire se ha senso che do anche la possibilita' di passare un dizionaro di configurazione
    # todo aggiungere baselines che possono fare cose simili, esperimento cold start, esperimento velocita di training


def run_sparsity_exp_mr(wandb_project, training_folds=tr_folds_mr, n_neg=100,
                        models=available_models, proportions_to_keep=(1, ), starting_seed=0, n_runs=30,
                        num_workers=os.cpu_count()):

    local_dataset_path_prefix = "./%s/datasets" % (wandb_project, )
    local_config_path_prefix = "./%s/configs" % (wandb_project, )
    local_model_path_prefix = "./%s/models" % (wandb_project, )
    local_result_path_prefix = "./%s/results" % (wandb_project, )

    # create folder to store datasets if it does not already exist
    if not os.path.exists(local_dataset_path_prefix):
        os.makedirs(local_dataset_path_prefix)

    # create folder to store configurations if it does not already exist
    if not os.path.exists(local_config_path_prefix):
        os.makedirs(local_config_path_prefix)

    # create folder to store models if it does not already exist
    if not os.path.exists(local_model_path_prefix):
        os.makedirs(local_model_path_prefix)

    # create folder to store results if it does not already exist
    if not os.path.exists(local_result_path_prefix):
        os.makedirs(local_result_path_prefix)

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
    Parallel(n_jobs=num_workers)(delayed(create_datasets_mr)(training_folds, seed, n_neg,
                                                             local_dataset_path_prefix, proportions_to_keep)
                                 for seed in range(starting_seed, starting_seed + n_runs))

    # run grid search of each model on each dataset with seed 0 and proportion 1
    Parallel(n_jobs=num_workers)(delayed(grid_search)(model,
                                                      "mr-%s-%d-%.2f-seed_%d" % (dataset_name, n_neg, p, starting_seed),
                                                      local_dataset_path_prefix,
                                                      local_config_path_prefix,
                                                      wandb_project)
                                 for model in models_list
                                 for dataset_name in training_folds
                                 for p in proportions_to_keep
                                 if check_compatibility(model, "mr-%s-%d-seed_%d" % (dataset_name, n_neg,
                                                                                     starting_seed)))

    # # train every model on every dataset with every seed with best configuration files obtained
    Parallel(n_jobs=num_workers)(delayed(train)(model,
                                                "mr-%s-%d-%.2f-seed_%d" % (dataset_name, n_neg, p, seed),
                                                starting_seed,
                                                local_dataset_path_prefix,
                                                local_config_path_prefix,
                                                local_model_path_prefix,
                                                wandb_project)
                                 for model in models_list
                                 for dataset_name in training_folds
                                 for seed in range(starting_seed, starting_seed + n_runs)
                                 for p in proportions_to_keep
                                 if check_compatibility(model, "mr-%s-%d-seed_%d" % (dataset_name, n_neg,
                                                                                     starting_seed)))
    # # #
    # # # # test every model on every dataset with every seed with best model files obtained
    Parallel(n_jobs=num_workers)(delayed(model_test)(model,
                                                     "mr-%s-%d-%.2f-seed_%d" % (dataset_name, n_neg, p, seed),
                                                     starting_seed,
                                                     local_dataset_path_prefix,
                                                     local_config_path_prefix,
                                                     local_model_path_prefix,
                                                     local_result_path_prefix,
                                                     wandb_project)
                                 for model in models_list
                                 for dataset_name in training_folds
                                 for seed in range(starting_seed, starting_seed + n_runs)
                                 for p in proportions_to_keep
                                 if check_compatibility(model, "mr-%s-%d-seed_%d" % (dataset_name, n_neg,
                                                                                     starting_seed)))
    create_report_json(local_result_path_prefix, "./%s" % (wandb_project,), seeds=list(range(starting_seed, n_runs)))
    generate_report_table("./%s/experiment-results.json" % (wandb_project,),
                          models=models, evaluation_modes=("mr", ), training_folds=training_folds,
                          proportions_to_keep=proportions_to_keep)

if __name__ == '__main__':
    run_experiment(wandb_project='final-stable-100-ndcg',
                   # evaluation_modes="ml\mr",
                   # training_folds=("ml(movies)|mr(genres)", "ml|mr(movies+genres)"),  # , "ml(movies)|mr(genres)"
                   # models="standard_mf",
                   # proportions_to_keep=(1, 0.5, 0.2, 0.1, 0.05, 0.01),  # 1, 0.5, 0.2, 0.1, 0.05
                   starting_seed=0, n_runs=30)
    # run_sparsity_exp_mr(wandb_project='sparsity-exp-mr',
    #                     proportions_to_keep=(1, 0.5, 0.2, 0.1, 0.05, 0.01),
    #                     starting_seed=0, n_runs=2)
    # todo fare anche il test con iper-parametri fissati per i modelli, provare con un numero variabile di fattori latenti
    # todo tipo testare a parita' di iper-parametri comuni come cambiano le performance
    # todo rifare esperimenti con aumento di sparsita'
    # todo capire perche' i generi non funzionano molto bene -> forse il metodo utilizzato non va benissimo, pensare a un modo migliore di integrare
    # todo fare esperimenti sparsita'
    # todo inventare qualcosa per cold-start
    # todo forse non funziona bene apprendere su un dataset e poi utilizzare su un altro
    # todo rifare l'esperimento con 100 come su mind reader
    # todo provare con meno seed ma fare grid search su ogni dataset
    # todo stesso modello con configurazione dei pesi diversa da risultati diversi -> era la normale utilizzata sul bias il problema
    # todo rifare anche esperimento con 200 negativi, per rendere le cose piu' difficili e vedere se la knowledge aiuta anche in quel caso
    # todo aggiungere i 20 seed all'esperimento nuovo
    # todo vedere se ottimizzando la ndcg le cose vanno meglio
    # todo andare bene a vedere cosa faccio sulla funzione che aumenta la sparsity, se riduco i rating di tutti gli utenti, poi succede che rendo le performance di LikesGenre basse
    # todo questo succede perche' gli embedding degli utenti di movielens avranno poca informazione al loro interno, ovvero non sapranno ancora se all'utente piace uno specifico genere o meno
    # todo mi aspetto che il caso in cui andra' meglio sara' quello movies+genres, anzi movies e basta perche' li ho veri rating degli utenti almeno per i film di MindReader e quindi anche di qualcuno in MovieLens
    # todo invece, nel caso dei generi rimane la problematica che sto togliendo troppe info dagli embedding degli utenti
    # todo secondo me, quella formula funziona bene su MindReader e basta, ovvero nel caso in cui diminuisco i rating sui film ma per gli utenti imparo bene i generi perche' ho tutti i rating sui generi
    # todo se io rendo sparso l'intero dataset, anche se ho i rating sui generi, gli embedding degli utenti di movielens saranno scarsi di informazione e quindi scarsa sara' la vericita' di LikesGenre, quindi
    # todo la formula non funzionera'
    # todo l'obiettivo e' far vedere le cose quando ho solo i rating di ml in test, senza nessun film anche in MindReader
    # todo si puo' pensare di frezzare i pesi di LikesGenre quando si fanno gli updates con la formula dei generi
    # todo capire se ha senso che i dataset sono diversi a seconda della valutazione a parita' di seed, ad esempio ml avra' piu' o meno ratings in training a seconda della valutazione, ad esempio, se un utente di ml non ha visto nessun film di mr, allora quel utente non sara' in validation o test in ml&mr
    # todo forse ml&mr e' il caso in cui vado peggio perche' ho molti meno esempi di validation/test rispetto agli altri due tipi di valutazione
    # todo sbilanciamento del dataset puo' essere un problema
    # todo e' un problema anche il fatto di aggregare voti su film con voti su generi forse? forse non dovrebbero influenzarsi in quella maniera
    # todo quello che voglio provare a fare ora e' utilizzare l'info dei generi di MindReader invece di quella di ml-100k come info di base
    # todo come allenare in maniera alternata? prima uno step sui film e poi frizzo e faccio uno step sui generi
    # gli dai un nome di file e ti va a cercare l'artefatto in locale, se c'e' non lo scarica, se no si
