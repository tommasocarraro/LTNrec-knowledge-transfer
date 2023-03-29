import torch.nn
import numpy as np
import wandb
from ltnrec.data import DataManager
from ltnrec.loaders import TrainingDataLoader, ValDataLoaderRatings, TrainingDataLoaderLTNClassification
from ltnrec.models import MatrixFactorization, MFTrainerClassifier, MatrixFactorizationLTN, LTNTrainerMFClassifier, \
    LTNTrainerMFClassifierTransferLearning
from torch.optim import Adam
from ltnrec.utils import set_seed
from ltnrec.models import FocalLossPyTorch
import pandas as pd
from sklearn.model_selection import train_test_split

data = DataManager("./datasets")
SEED = 0
TR_BATCH_SIZE = 256
VAL_BATCH_SIZE = 256

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
            'p_sat_agg': {"values": [1, 2, 3, 4]},
            'p_pos': {"values": [1, 2, 3, 4]},
            'p_neg': {"values": [1, 2, 3, 4]}
        }
    }

SWEEP_CONFIG_LTN_TRANSFER = {
        'method': "bayes",
        'metric': {'goal': 'maximize', 'name': 'fbeta-1.0'},
        'parameters': {
            'k': {"values": [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]},  # [1, 16, 32, 64, 128, 256, 512]
            'init_std': {"values": [0.01, 0.001, 0.0001]},
            'lr': {"distribution": "log_uniform_values", "min": 0.0001, "max": 0.1},
            'wd': {"distribution": "log_uniform_values", "min": 0.00001, "max": 0.1},
            'biased': {"values": [1, 0]},
            'p_sat_agg': {"values": [1, 2, 3, 4]},
            'p_pos': {"values": [1, 2, 3, 4]},
            'p_neg': {"values": [1, 2, 3, 4]},
            'p_forall': {"values": [1, 2, 3, 4]},
            'p_exists': {"values": [1, 2, 3, 4]},
            'binary_likes_genre': {"values": [0, 1]}
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


def likes_training(train_set, val_set, config, ltn=False, genres=False, get_genre_matrix=False):
    """
    It performs the training of the LikesGenre (if genre==True) or Likes (if genre==False) predicate
    using the given training and validation set. The validation set is used to perform early stopping and prevent
    overfitting.

    :param train_set: train set on which the tuning is performed
    :param val_set: validation set on which the tuning is evaluated
    :param config: configuration dictionary containing the hyper-parameter values to train the model
    :param ltn: whether it has to be used LTN to perform the tuning or classic Matrix Factorization
    :param genres: whether the tuning has to be performed for the LikesGenre (genres==True) or Likes
    (genres==False) predicate
    :param get_genre_matrix: whether a user x genres pre-trained matrix has to be returned or not
    :param
    """
    # create loaders for training and validating
    if ltn:
        train_loader = TrainingDataLoaderLTNClassification(train_set["ratings"], batch_size=TR_BATCH_SIZE)
    else:
        train_loader = TrainingDataLoader(train_set["ratings"], TR_BATCH_SIZE)

    val_loader = ValDataLoaderRatings(val_set["ratings"], VAL_BATCH_SIZE)

    # get proportion of positive examples in the dataset

    pos_prop = get_pos_prop(train_set["ratings"])

    # define function for computing user x genre matrix

    def compute_u_g_matrix(mf):
        matrix = torch.matmul(mf.u_emb.weight.data, torch.t(mf.i_emb.weight.data))
        if mf.biased:
            matrix = torch.add(matrix, mf.u_bias.weight.data)
            matrix = torch.add(matrix, torch.t(mf.i_bias.weight.data))
        # apply sigmoid to predictions
        return torch.sigmoid(matrix)

    # define function to call for performing the training for the standard Matrix Factorization

    def train_likes_standard():
        mf = MatrixFactorization(n_users, n_items if not genres else n_genres, config["k"], config["biased"],
                                 init_std=config["init_std"])
        optimizer = Adam(mf.parameters(), lr=config["lr"], weight_decay=config["wd"])
        trainer = MFTrainerClassifier(mf, optimizer,
                                      loss=FocalLossPyTorch(alpha=(1. - pos_prop) if config["alpha"] else -1.,
                                                            gamma=config["gamma"]),
                                      wandb_train=False, threshold=0.5)
        trainer.train(train_loader, val_loader, "fbeta-1.0", n_epochs=1000, early=10, verbose=1,
                      save_path="likes_genre_standard.pth" if genres else "likes_standard.pth")

        if get_genre_matrix:
            trainer.load_model("likes_genre_standard.pth")
            # compute and return matrix
            return compute_u_g_matrix(mf)

    # define function to call for performing the training for the Matrix Factorization inside the LTN framework

    def train_likes_ltn():
        mf = MatrixFactorizationLTN(n_users, n_items if not genres else n_genres, config["k"], config["biased"],
                                    config["init_std"], normalize=True)
        optimizer = Adam(mf.parameters(), lr=config["lr"], weight_decay=config["wd"])
        trainer = LTNTrainerMFClassifier(mf, optimizer, p_pos=config["p_pos"], p_neg=config["p_neg"],
                                         p_sat_agg=config["p_sat_agg"],
                                         wandb_train=False, threshold=0.5)
        trainer.train(train_loader, val_loader, "fbeta-1.0", n_epochs=1000, early=10, verbose=1,
                      save_path="likes_genre_ltn.pth" if genres else "likes_ltn.pth")

        if get_genre_matrix:
            trainer.load_model("likes_genre_ltn.pth")
            # compute and return matrix
            return compute_u_g_matrix(mf)

    # launch the training of the LikesGenre predicate

    # set seed for training
    set_seed(SEED)

    if ltn:
        u_g_matrix = train_likes_ltn()
    else:
        u_g_matrix = train_likes_standard()

    if get_genre_matrix:
        return u_g_matrix


def likes_tuning(train_set, val_set, ltn=False, genres=False):
    """
    It performs the hyper-parameter tuning of the LikesGenre (if genre==True) or Likes (if genre==False) predicate
    using the given training and validation set.

    :param train_set: train set on which the tuning is performed
    :param val_set: validation set on which the tuning is evaluated
    :param ltn: whether it has to be used LTN to perform the tuning or classic Matrix Factorization
    :param genres: whether the tuning has to be performed for the LikesGenre (genres==True) or Likes
    (genres==False) predicate
    :param
    """
    # create loaders for training and validating
    if ltn:
        train_loader = TrainingDataLoaderLTNClassification(train_set["ratings"], batch_size=TR_BATCH_SIZE)
    else:
        train_loader = TrainingDataLoader(train_set["ratings"], TR_BATCH_SIZE)

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

            mf = MatrixFactorization(n_users, n_items if not genres else n_genres, k, biased, init_std=init_std)
            optimizer = Adam(mf.parameters(), lr=lr, weight_decay=wd)
            trainer = MFTrainerClassifier(mf, optimizer,
                                          loss=FocalLossPyTorch(alpha=(1. - pos_prop) if alpha else -1., gamma=gamma),
                                          wandb_train=True, threshold=0.5)
            trainer.train(train_loader, val_loader, "fbeta-1.0", n_epochs=1000, early=10, verbose=1)

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

            mf = MatrixFactorizationLTN(n_users, n_items if not genres else n_genres, k, biased, init_std,
                                        normalize=True)
            optimizer = Adam(mf.parameters(), lr=lr, weight_decay=wd)
            trainer = LTNTrainerMFClassifier(mf, optimizer, p_pos=p_pos, p_neg=p_neg, p_sat_agg=p_sat_agg,
                                             wandb_train=True, threshold=0.5)
            trainer.train(train_loader, val_loader, "fbeta-1.0", n_epochs=1000, early=10, verbose=1)

    # launch the WandB sweep for the LikesGenre predicate

    sweep_id = wandb.sweep(sweep=SWEEP_CONFIG if not ltn else SWEEP_CONFIG_LTN,
                           project=("likes-genre-standard" if genres else "likes-standard")
                           if not ltn else ("likes-genre-ltn" if genres else "likes-ltn"))
    wandb.agent(sweep_id, function=tune_likes_standard if not ltn else tune_likes_ltn, count=100)


def final_model_tuning(train_set, val_set):
    """
    It performs the hyper-parameter tuning of the final model. This model has a formula that forces the latent factors
    to produce the ground truth (target ratings), and a formula which acts as a kind of regularization for the latent
    factors. This second formula performs knowledge transfer and transfer learning. Based on some learned preferences
    about movie genres, we should be able to increase the performance of a model learnt to classify movie preferences.
    The performance should increase more when the dataset is made more challenging by increasing its sparsity.

    :param train_set: train set on which the tuning is performed
    :param val_set: validation set on which the tuning is evaluated
    :param
    """
    # create loaders for training and validating
    train_loader = TrainingDataLoaderLTNClassification(train_set["ratings"], non_relevant_sampling=True,
                                                       n_users=n_users, n_items=n_items, batch_size=TR_BATCH_SIZE)
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
            p_forall = wandb.config.p_forall
            p_exists = wandb.config.p_exists
            binary_likes_genre = wandb.config.binary_likes_genre

            mf = MatrixFactorizationLTN(n_users, n_items, k, biased, init_std, normalize=True)
            optimizer = Adam(mf.parameters(), lr=lr, weight_decay=wd)

            # filter user x genres matrix if requested
            if binary_likes_genre:
                filtered_u_g_matrix = torch.where(user_genres_matrix >= 0.5, 1., 0.)
            else:
                filtered_u_g_matrix = user_genres_matrix

            trainer = LTNTrainerMFClassifierTransferLearning(mf, optimizer,
                                                             u_g_matrix=filtered_u_g_matrix,
                                                             i_g_matrix=item_genres_matrix, p_pos=p_pos,
                                                             p_neg=p_neg,
                                                             p_sat_agg=p_sat_agg, p_forall=p_forall,
                                                             p_exists=p_exists,
                                                             wandb_train=True, threshold=0.5)

            trainer.train(train_loader, val_loader, "fbeta-1.0", n_epochs=1000, early=10, verbose=1)

    # launch the WandB sweep for the Likes predicate with transfer learning

    sweep_id = wandb.sweep(sweep=SWEEP_CONFIG_LTN_TRANSFER, project="transfer-learning")
    wandb.agent(sweep_id, function=tune_model, count=300)


def final_model_training(train_set, val_set, config):
    """
    It performs the training of the final model on the given dataset with the given configurtion of hyper-parameters.

    :param train_set: train set on which the training is performed
    :param val_set: validation set on which the model is evaluated
    :param config: dictionary containing the configuration of hyper-parameter for learning the model
    """
    # create loaders for training and validating
    train_loader = TrainingDataLoaderLTNClassification(train_set["ratings"], non_relevant_sampling=True,
                                                       n_users=n_users, n_items=n_items, batch_size=TR_BATCH_SIZE)
    val_loader = ValDataLoaderRatings(val_set["ratings"], VAL_BATCH_SIZE)

    mf = MatrixFactorizationLTN(n_users, n_items, config["k"], config["biased"], config["init_std"], normalize=True)
    optimizer = Adam(mf.parameters(), lr=config["lr"], weight_decay=config["wd"])

    # filter user X genres matrix, if requested
    if config["binary_likes_genre"]:
        filtered_u_g_matrix = torch.where(user_genres_matrix >= 0.5, 1., 0.)
    else:
        filtered_u_g_matrix = user_genres_matrix

    trainer = LTNTrainerMFClassifierTransferLearning(mf, optimizer,
                                                     u_g_matrix=filtered_u_g_matrix,
                                                     i_g_matrix=item_genres_matrix, p_pos=config["p_pos"],
                                                     p_neg=config["p_neg"],
                                                     p_sat_agg=config["p_sat_agg"], p_forall=config["p_forall"],
                                                     p_exists=config["p_exists"],
                                                     wandb_train=False, threshold=0.5)
    # set seed for training
    set_seed(SEED)
    trainer.train(train_loader, val_loader, "fbeta-1.0", n_epochs=1000, early=10, verbose=1,
                  save_path="final_model.pth")


if __name__ == "__main__":

    # step 0: get data for the entire experiment (data for genre and movie preferences)

    n_users, n_genres, n_items, genre_folds, movie_folds, item_genres_matrix = data.get_mr_200k_dataset(seed=0,
                                                                                                        val_mode="rating-prediction",
                                                                                                        movie_val_size=0.1,
                                                                                                        movie_test_size=0.2)
    g_train_set, g_val_set = genre_folds.values()
    u_i_matrix, train_set, train_set_small, val_set, test_set = movie_folds.values()

    # step 1: hyper-parameter tuning of the LikesGenre predicate (on genres) (both MF and LTN models)

    # likes_tuning(g_train_set, g_val_set, ltn=False, genres=True)
    # likes_tuning(g_train_set, g_val_set, ltn=True, genres=True)

    # step 1.1: prepare dictionaries with best configurations found for the two LikesGenre models

    BEST_MF_GENRE = {"biased": 1, "k": 50, "alpha": 1, "gamma": 3, "init_std": 0.0001, "lr": 0.001, "wd": 0.00001}

    BEST_LTN_GENRE = {"biased": 1, "k": 50, "init_std": 0.0001, "lr": 0.0005, "p_neg": 3, "p_pos": 4, "p_sat_agg": 3,
                      "wd": 0.0001}

    # step 1.2: train the two models with the found configuration and see the results

    # likes_training(g_train_set, g_val_set, BEST_MF_GENRE, ltn=False, genres=True)  # MF reaches Epoch 35 - Train loss 0.004 - Val fbeta-1.0 0.573 - Neg prec 0.553 - Pos prec 0.894 - Neg rec 0.589 - Pos rec 0.879 - Neg f 0.570 - Pos f 0.887 - tn 527 - fp 368 - fn 426 - tp 3105
    # likes_training(g_train_set, g_val_set, BEST_LTN_GENRE, ltn=True, genres=True)  # LTN reaches Epoch 46 - Train loss 0.341 - Val fbeta-1.0 0.566 - training_overall_sat 0.659 - pos_sat 0.645 - neg_sat 0.675 - Neg prec 0.505 - Pos prec 0.901 - Neg rec 0.635 - Pos rec 0.843 - Neg f 0.563 - Pos f 0.871 - tn 568 - fp 327 - fn 556 - tp 2975

    # step 2: hyper-parameter tuning of the Likes predicate (on movies) (both MF and LTN)

    # likes_tuning(train_set_small, val_set, ltn=False, genres=False)
    # likes_tuning(train_set_small, val_set, ltn=True, genres=False)

    # step 2.1: prepare dictionaries with best configurations found for the two Likes models

    BEST_MF = {"biased": 1, "k": 50, "alpha": 1, "gamma": 0, "init_std": 0.0001, "lr": 0.001, "wd": 0.00005}

    BEST_LTN = {"biased": 1, "k": 50, "init_std": 0.0001, "lr": 0.005, "p_neg": 3, "p_pos": 3, "p_sat_agg": 4,
                "wd": 0.00005}

    # step 2.2: train the two models with the found configuration and see the results

    # likes_training(train_set_small, val_set, BEST_MF, ltn=False, genres=False)  # MF reaches Epoch 39 - Train loss 0.213 - Val fbeta-1.0 0.648 - Neg prec 0.603 - Pos prec 0.816 - Neg rec 0.698 - Pos rec 0.745 - Neg f 0.647 - Pos f 0.779 - tn 1938 - fp 840 - fn 1277 - tp 3732
    # likes_training(train_set_small, val_set, BEST_LTN, ltn=True, genres=False)  # LTN reaches Epoch 5 - Train loss 0.429 - Val fbeta-1.0 0.650 - training_overall_sat 0.571 - pos_sat 0.573 - neg_sat 0.570 - Neg prec 0.607 - Pos prec 0.817 - Neg rec 0.697 - Pos rec 0.750 - Neg f 0.649 - Pos f 0.782 - tn 1937 - fp 841 - fn 1254 - tp 3755

    # step 3: hyper-parameter tuning of the complete LTN model with the best LikesGenre model pre-trained.
    # This model will try to learn an LTN MF model with the addition of some background knowledge on the
    # genre preferences (if a user does not like a genre, than the user should not like movies with that genre)

    # get user x genres matrix

    user_genres_matrix = likes_training(g_train_set, g_val_set, BEST_LTN_GENRE, ltn=True, genres=True,
                                        get_genre_matrix=True)

    # final_model_tuning(train_set_small, val_set)

    # step 3.1: prepare dictionary with configuration of best hyper-parameters

    BEST_LTN_TRANSFER = {"biased": 1, "k": 50, "init_std": 0.0001, "lr": 0.001, "p_neg": 3, "p_pos": 3, "p_sat_agg": 1,
                         "wd": 0.00005, "p_forall": 4, "p_exists": 3, "binary_likes_genre": 1}

    # step 3.2: train the model with the best hyper-parameters

    # final_model_training(train_set_small, val_set, BEST_LTN_TRANSFER)  # with k=200 we have better performance but it
    # is not a big gap, we can keep k=50 as the other models

    # step 4: compare the performance of the different models on the test set

    # compare_performance_on_test()  # todo implementare il codice per il test del modello, simile a quello per la validation

    # step 5: increase sparsity of the dataset and re-train the model at step 3 with same hyper-parameters on
    # datasets with different sparsity to see if the knowledge about the genres helps a little bit
    # for each dataset, we need to train both the model without genres and the model with genres and compare
    # the differences

    train_set_02 = increase_data_sparsity(train_set_small, 0.3, 2)
    print("\n ----------- Begin LTN without formula --------------- \n")
    likes_training(train_set_02, val_set, BEST_LTN, ltn=True, genres=False)
    print("\n ----------- Begin LTN with formula ------------------ \n")
    final_model_training(train_set_02, val_set, BEST_LTN_TRANSFER)


    # todo forse, non facendolo user-level, la valutazione e' ancora piu' fair perche' togliamo bias dagli utenti
    # todo sarebbe utile anche fare un esperimento per il cold-start - pero' quello potrebbe essere un paper successivo
    # todo serve fare l'esperimento con piu' seed diversi che cambiano il dataset, per vedere come il modello si
    #  comporta
    # todo hyper-parameter tuning teniamo quella sul seed 0
