import torch.nn
import numpy as np
import wandb
from ltnrec.data import DataManager
from ltnrec.loaders import TrainingDataLoader, ValDataLoaderRatings, TrainingDataLoaderLTNClassification, ValDataLoaderRanking
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
            'p_forall_f1': {"values": [1, 2, 3, 4]},
            'p_forall_f2': {"values": [1, 2, 3, 4]},
            'p_in_f1': {"values": [1, 2, 3, 4]},
            'p_in_f2': {"values": [1, 2, 3, 4]},
            'forall_f1': {"values": [0, 1]},
            'forall_f2': {"values": [0, 1]},
            'binary_likes_genre': {"values": [0, 1]},
            'f2': {'values': [0, 1]}
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


def likes_training(train_set, val_set, config, metric, ltn=False, genres=False, get_genre_matrix=False,
                   test_loader=None, just_load=False):
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

    :param train_set: train set on which the tuning is performed
    :param val_set: validation set on which the tuning is evaluated
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
    :param
    """
    # create loaders for training and validating
    if ltn:
        train_loader = TrainingDataLoaderLTNClassification(train_set["ratings"], batch_size=TR_BATCH_SIZE)
    else:
        train_loader = TrainingDataLoader(train_set["ratings"], TR_BATCH_SIZE)

    if "@" in metric or metric == "auc":
        val_loader = ValDataLoaderRanking(val_set, VAL_BATCH_SIZE)
    else:
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
        if not just_load:
            trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1,
                          save_path="likes_genre_standard.pth" if genres else "likes_standard.pth")

        if get_genre_matrix:
            trainer.load_model("likes_genre_standard.pth")
            # compute and return matrix
            return compute_u_g_matrix(mf)

        if test_loader is not None:
            trainer.load_model("likes_standard.pth")
            return trainer.test(test_loader, "rating-prediction", fbeta=metric)

    # define function to call for performing the training for the Matrix Factorization inside the LTN framework

    def train_likes_ltn():
        mf = MatrixFactorizationLTN(n_users, n_items if not genres else n_genres, config["k"], config["biased"],
                                    config["init_std"], normalize=True)
        optimizer = Adam(mf.parameters(), lr=config["lr"], weight_decay=config["wd"])
        trainer = LTNTrainerMFClassifier(mf, optimizer, p_pos=config["p_pos"], p_neg=config["p_neg"],
                                         p_sat_agg=config["p_sat_agg"],
                                         wandb_train=False, threshold=0.5)
        if not just_load:
            trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1,
                          save_path="likes_genre_ltn.pth" if genres else "likes_ltn.pth")

        if get_genre_matrix:
            trainer.load_model("likes_genre_ltn.pth")
            # compute and return matrix
            return compute_u_g_matrix(mf)

        if test_loader is not None:
            trainer.load_model("likes_ltn.pth")
            if "@" in metric:
                m = "1+random"
            elif metric == "auc":
                m = "auc"
            else:
                m = "rating-prediction"

            return trainer.test(test_loader, m, fbeta=metric)

    # launch the training of the LikesGenre predicate

    # set seed for training
    set_seed(SEED)

    if ltn:
        out = train_likes_ltn()
    else:
        out = train_likes_standard()

    return out


def likes_tuning(train_set, val_set, metric, ltn=False, genres=False):
    """
    It performs the hyper-parameter tuning of the LikesGenre (if genre==True) or Likes (if genre==False) predicate
    using the given training and validation set.

    :param train_set: train set on which the tuning is performed
    :param val_set: validation set on which the tuning is evaluated
    :param metric: validation metric that has to be used
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

            mf = MatrixFactorization(n_users, n_items if not genres else n_genres, k, biased, init_std=init_std)
            optimizer = Adam(mf.parameters(), lr=lr, weight_decay=wd)
            trainer = MFTrainerClassifier(mf, optimizer,
                                          loss=FocalLossPyTorch(alpha=(1. - pos_prop) if alpha else -1., gamma=gamma),
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

            mf = MatrixFactorizationLTN(n_users, n_items if not genres else n_genres, k, biased, init_std,
                                        normalize=True)
            optimizer = Adam(mf.parameters(), lr=lr, weight_decay=wd)
            trainer = LTNTrainerMFClassifier(mf, optimizer, p_pos=p_pos, p_neg=p_neg, p_sat_agg=p_sat_agg,
                                             wandb_train=True, threshold=0.5)
            trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1)

    # launch the WandB sweep for the LikesGenre predicate

    SWEEP_CONFIG['metric']['name'] = metric
    SWEEP_CONFIG_LTN['metric']['name'] = metric

    sweep_id = wandb.sweep(sweep=SWEEP_CONFIG if not ltn else SWEEP_CONFIG_LTN,
                           project=("likes-genre-standard" if genres else "likes-standard")
                           if not ltn else ("likes-genre-ltn" if genres else "likes-ltn"))
    wandb.agent(sweep_id, function=tune_likes_standard if not ltn else tune_likes_ltn, count=100)


def final_model_tuning(train_set, val_set, metric, sweep_id=None):
    """
    It performs the hyper-parameter tuning of the final model. This model has a formula that forces the latent factors
    to produce the ground truth (target ratings), and a formula which acts as a kind of regularization for the latent
    factors. This second formula performs knowledge transfer and transfer learning. Based on some learned preferences
    about movie genres, we should be able to increase the performance of a model learnt to classify movie preferences.
    The performance should increase more when the dataset is made more challenging by increasing its sparsity.

    :param train_set: train set on which the tuning is performed
    :param val_set: validation set on which the tuning is evaluated
    :param metric: validation metric
    :param sweep_id: id of WandB sweep, to be used if a sweep has been interrupted and it is needed to continue it.
    :param
    """
    # create loaders for training and validating
    train_loader = TrainingDataLoaderLTNClassification(train_set["ratings"], non_relevant_sampling=True,
                                                       n_users=n_users, n_items=n_items, batch_size=TR_BATCH_SIZE)
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
                                                             p_sat_agg=p_sat_agg, p_forall_f1=p_forall_f1,
                                                             p_in_f1=p_in_f1, forall_f1=forall_f1, f2=f2,
                                                             p_forall_f2=p_forall_f2, p_in_f2=p_in_f2,
                                                             forall_f2=forall_f2, wandb_train=True, threshold=0.5)

            trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1)

    # launch the WandB sweep for the Likes predicate with transfer learning

    SWEEP_CONFIG_LTN_TRANSFER['metric']['name'] = metric

    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=SWEEP_CONFIG_LTN_TRANSFER, project="transfer-learning")
    wandb.agent(sweep_id, function=tune_model, count=150, project="transfer-learning")


def final_model_training(train_set, val_set, config, metric, test_loader=None):
    """
    It performs the training of the final model on the given dataset with the given configuration of hyper-parameters.

    If test_loader is not None, it tests the best model found on the validation set on the test set.

    :param train_set: train set on which the training is performed
    :param val_set: validation set on which the model is evaluated
    :param config: dictionary containing the configuration of hyper-parameter for learning the model
    :param metric: metric used to validate and test the model
    :param test_loader: test loader to test the performance of the model on the test set. Defaults to None
    """
    # create loaders for training and validating
    train_loader = TrainingDataLoaderLTNClassification(train_set["ratings"], non_relevant_sampling=True,
                                                       n_users=n_users, n_items=n_items, batch_size=TR_BATCH_SIZE)
    if "@" in metric or metric == "auc":
        val_loader = ValDataLoaderRanking(val_set, VAL_BATCH_SIZE)
    else:
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
                                                     p_sat_agg=config["p_sat_agg"], p_forall_f1=config["p_forall_f1"],
                                                     p_in_f1=config["p_in_f1"], forall_f1=config["forall_f1"],
                                                     f2=config["f2"], p_forall_f2=config["p_forall_f2"],
                                                     p_in_f2=config["p_in_f2"], forall_f2=config["forall_f2"],
                                                     wandb_train=False, threshold=0.5)
    # set seed for training
    set_seed(SEED)
    trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1,
                  save_path="final_model.pth")

    if test_loader is not None:
        trainer.load_model("final_model.pth")
        if "@" in metric:
            m = "1+random"
        elif metric == "auc":
            m = "auc"
        else:
            m = "rating-prediction"

        return trainer.test(test_loader, m, fbeta=metric)


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

    # likes_tuning(train_set_small, val_set, ltn=False, genres=False, metric="auc")
    # likes_tuning(train_set_small, val_set, ltn=True, genres=False, metric="auc")
    likes_tuning(train_set_small, val_set, ltn=False, genres=False, metric="acc")
    likes_tuning(train_set_small, val_set, ltn=True, genres=False, metric="acc")

    np.d()

    # step 2.1: prepare dictionaries with best configurations found for the two Likes models

    BEST_MF_FBETA = {"biased": 1, "k": 50, "alpha": 1, "gamma": 0, "init_std": 0.0001, "lr": 0.001, "wd": 0.00005}

    BEST_LTN_FBETA = {"biased": 1, "k": 50, "init_std": 0.0001, "lr": 0.005, "p_neg": 3, "p_pos": 3, "p_sat_agg": 4,
                      "wd": 0.00005}

    BEST_MF_AUC = {"biased": 1, "k": 500, "alpha": 0, "gamma": 3, "init_std": 0.0001, "lr": 0.0005, "wd": 0.00001}

    BEST_LTN_AUC = {"biased": 1, "k": 300, "init_std": 0.0001, "lr": 0.005, "p_neg": 4, "p_pos": 4, "p_sat_agg": 4,
                    "wd": 0.00005}

    # step 2.2: train the two models with the found configuration and see the results

    # likes_training(train_set_small, val_set, BEST_MF, ltn=False, genres=False)  # MF reaches Epoch 39 - Train loss 0.213 - Val fbeta-1.0 0.648 - Neg prec 0.603 - Pos prec 0.816 - Neg rec 0.698 - Pos rec 0.745 - Neg f 0.647 - Pos f 0.779 - tn 1938 - fp 840 - fn 1277 - tp 3732
    # likes_training(train_set_small, val_set, BEST_LTN, ltn=True, genres=False)  # LTN reaches Epoch 5 - Train loss 0.429 - Val fbeta-1.0 0.650 - training_overall_sat 0.571 - pos_sat 0.573 - neg_sat 0.570 - Neg prec 0.607 - Pos prec 0.817 - Neg rec 0.697 - Pos rec 0.750 - Neg f 0.649 - Pos f 0.782 - tn 1937 - fp 841 - fn 1254 - tp 3755
    # likes_training(train_set_small, val_set, BEST_MF_AUC, ltn=False, genres=False, metric="auc")  # MF reaches Epoch 59 - Train loss 0.015 - Val auc 0.776
    # likes_training(train_set_small, val_set, BEST_LTN_AUC, ltn=True, genres=False, metric="auc")  # LTN reaches Epoch 16 - Train loss 0.389 - Val auc 0.775 - training_overall_sat 0.611 - pos_sat 0.611 - neg_sat 0.613

    # todo osservazione -> la metrica su LTN e' molto piu' ballerina e mi da l'idea di aver ottenuto quel punteggio solo by chance perche' ha fatto un mega salto e non ha piu' raggiunto un punteggio simile durante il training

    # step 3: hyper-parameter tuning of the complete LTN model with the best LikesGenre model pre-trained.
    # This model will try to learn an LTN MF model with the addition of some background knowledge on the
    # genre preferences (if a user does not like a genre, than the user should not like movies with that genre)

    # get user x genres matrix

    user_genres_matrix = likes_training(g_train_set, g_val_set, BEST_LTN_GENRE, metric="fbeta-1.0", ltn=True,
                                        genres=True, get_genre_matrix=True, just_load=True)

    # final_model_tuning(train_set_small, val_set, metric="auc", sweep_id="b9faocvt")

    # step 3.1: prepare dictionary with configuration of best hyper-parameters

    BEST_LTN_TRANSFER_FBETA = {"biased": 1, "k": 50, "init_std": 0.0001, "lr": 0.001, "p_neg": 3, "p_pos": 3, "p_sat_agg": 1,
                               "wd": 0.00005, "p_forall": 4, "p_exists": 3, "binary_likes_genre": 1, "pos_transfer": 0,
                               "p_pos_f_in": 2, "p_pos_f_out": 2}

    BEST_LTN_TRANSFER_AUC = {"biased": 1, "k": 20, "init_std": 0.0001, "lr": 0.001, "p_neg": 2, "p_pos": 4,
                             "p_sat_agg": 2, "wd": 0.00005, "p_forall_f1": 4, "p_in_f1": 4, "forall_f1": 1, "f2": 1,
                             "p_forall_f2": 1, "p_in_f2": 1, "forall_f2": 1, "binary_likes_genre": 1}

    # step 3.2: train the model with the best hyper-parameters

    # final_model_training(train_set_small, val_set, BEST_LTN_TRANSFER_AUC, metric="auc")  # with k=200 we have better performance but it
    # is not a big gap, we can keep k=50 as the other models

    # step 4: compare the performance of the different models on the test set

    # compare_performance_on_test()  # todo implementare il codice per il test del modello, simile a quello per la validation

    # step 5: increase sparsity of the dataset and re-train the model at step 3 with same hyper-parameters on
    # datasets with different sparsity to see if the knowledge about the genres helps a little bit
    # for each dataset, we need to train both the model without genres and the model with genres and compare
    # the differences

    # create test loader

    # test_loader = ValDataLoaderRatings(test_set["ratings"], VAL_BATCH_SIZE)
    test_loader = ValDataLoaderRanking(test_set, VAL_BATCH_SIZE)

    train_set_02 = increase_data_sparsity(train_set_small, 0.01, SEED)
    # print("\n ----------- Begin LTN without formula --------------- \n")
    # print(likes_training(train_set_02, val_set, BEST_LTN_AUC, metric="auc", ltn=True, genres=False,
    #                      test_loader=test_loader))
    # print("\n ----------- Begin LTN with formula ------------------ \n")
    # print(final_model_training(train_set_02, val_set, BEST_LTN_TRANSFER_AUC, metric="auc", test_loader=test_loader))

    # try with tuning and see if we are able to get massive improvements or not
    likes_tuning(train_set_02, val_set, metric="auc", ltn=True, genres=False)
    final_model_tuning(train_set_02, val_set, "auc")

    # todo provare anche con mse perche' funziona meglio per ranking, perche' distacca il decision boundary
    # todo se invece devo fare classificazione, va bene cosi e uso fbeta o acc, devo solo decidere cosa voglio fare


    # todo forse, non facendolo user-level, la valutazione e' ancora piu' fair perche' togliamo bias dagli utenti
    # todo sarebbe utile anche fare un esperimento per il cold-start - pero' quello potrebbe essere un paper successivo
    # todo serve fare l'esperimento con piu' seed diversi che cambiano il dataset, per vedere come il modello si
    #  comporta
    # todo hyper-parameter tuning teniamo quella sul seed 0
    # todo forse bisogna aggiungere anche l'esperimento che mette la formula anche sui positivi
    # todo bisognare fare tuning o diminuire il numero di fattori latenti
    # todo la fbeta e' meglio ma non mi spiego il perche', va bene sulla classe negativa ma male sulla positiva.
    #  Probabilmente perche' quella formula abbassa i negativi e non fa niente per i positivi. Abbiamo pochi positivi
    #  per generalizzare e quindi serve anche la formula positiva, magari con un per ogni
    # todo dovrei fare tuning su ogni piccolo dataset piu' sparso, per essere sicuro che sia tutto ok
    # todo provare a usare l'accuracy per validare il modello perche' abbiamo le classi che sono quasi bilanciate
    # todo forse mse e f1 sono metriche piu' appropriate e piu' stabili perche' meno sensibili al sampling, su mse e f1 si pescano molti piu' esempi rispetto a uno per utente


    # {'fbeta-1.0': {'fbeta-1.0': 0.519609794943261, 'neg_prec': 0.3970185579555826, 'pos_prec': 0.7272151898734177, 'neg_rec': 0.7517281105990783, 'pos_rec': 0.36697540721814115, 'neg_f': 0.519609794943261, 'pos_f': 0.4877945234557418, 'tn': 2610, 'fp': 862, 'fn': 3964, 'tp': 2298, 'sensitivity': 0.36697540721814115, 'specificity': 0.7517281105990783, 'acc': 0.5042120402712142}, 'mse': 0.4957879597287857, 'rmse': 0.7041221198973838}
