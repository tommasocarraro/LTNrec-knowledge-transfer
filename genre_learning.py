import torch.nn
import numpy as np
import wandb
from ltnrec.data import DataManager
from ltnrec.loaders import TrainingDataLoader, BalancedTrainingDataLoader, ValDataLoaderMSE, \
    BalancedTrainingDataLoaderNeg
from ltnrec.models import MatrixFactorization, MFTrainer, DeepMatrixFactorization
from torch.optim import Adam
from ltnrec.utils import set_seed
from ltnrec.models import FocalLoss
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight


# mse = torch.nn.MSELoss()
# mse3 = lambda pred, gt: torch.mean(torch.pow(pred - gt, 3))
# # cross entropy loss
# ce = torch.nn.BCELoss()
# # focal loss
# focal = FocalLoss(alpha=0.1, gamma=3, reduction="sum")
# # focal regression
# focal_regr = FocalLossRegression(alpha=0.1, gamma=3, beta=0.5, reduction="mean")
# # weighted mse loss
# wmse = lambda pred, gt, w: torch.mean(w * torch.square(pred - gt))
# # shrinkage loss
# sh = lambda pred, gt: torch.mean(torch.square(pred - gt) /
#                                       (1 + torch.exp(10 * (0.2 - (1 - torch.exp(-torch.abs(pred - gt)))))))

data = DataManager("./datasets")
SEED = 0


def try_conf():
    with wandb.init():
        set_seed(SEED)
        k = wandb.config.k
        biased = wandb.config.biased
        lr = wandb.config.lr
        wd = wandb.config.wd
        tr_batch_size = wandb.config.tr_batch_size
        k_filter = wandb.config.k_filter
        genre_threshold = wandb.config.genre_threshold
        balanced = wandb.config.balanced
        init_std = wandb.config.init_std
        dropout = wandb.config.dropout

        n_users, n_genres, genre_folds = data.get_mr_200k_genre_ratings(SEED, genre_threshold=genre_threshold,
                                                                        k_filter=k_filter, genre_val_size=0.2)
        train_set, val_set = genre_folds.values()
        val_loader = ValDataLoaderMSE(val_set["ratings"], batch_size=256)
        if balanced:
            train_loader = BalancedTrainingDataLoader(train_set["ratings"], tr_batch_size)
        else:
            train_loader = TrainingDataLoader(train_set["ratings"], tr_batch_size)
        mf = MatrixFactorization(n_users, n_genres, k, biased, init_std, dropout)
        optimizer = Adam(mf.parameters(), lr=lr, weight_decay=wd)
        trainer = MFTrainer(mf, optimizer)
        trainer.train(train_loader, val_loader, None, n_epochs=1000, early=20, verbose=1, wandb_train=True,
                      val_loss_early=True)


def try_conf_sig():
    with wandb.init():
        k = wandb.config.k
        biased = wandb.config.biased
        lr = wandb.config.lr
        wd = wandb.config.wd
        tr_batch_size = wandb.config.tr_batch_size
        init_std = wandb.config.init_std
        dropout = wandb.config.dropout
        alpha = wandb.config.alpha
        gamma = wandb.config.gamma
        reduction = wandb.config.reduction
        threshold = wandb.config.threshold

        n_users, n_genres, genre_folds, _ = data.get_mr_200k_genre_ratings(SEED, genre_threshold=None,
                                                                        k_filter=None, genre_val_size=0.2,
                                                                        user_level_split=False, binary_ratings=True)
        train_set, val_set = genre_folds.values()
        val_loader = ValDataLoaderMSE(val_set["ratings"], batch_size=256)
        train_loader = TrainingDataLoader(train_set["ratings"], tr_batch_size)
        mf = MatrixFactorization(n_users, n_genres, k, biased, init_std, dropout, normalize=True)
        optimizer = Adam(mf.parameters(), lr=lr, weight_decay=wd)
        trainer = MFTrainer(mf, optimizer, wandb_train=True, beta=0.5, loss=FocalLoss(alpha, gamma, reduction),
                            threshold=threshold)
        trainer.train(train_loader, val_loader, "f-beta", n_epochs=1000, early=30, verbose=1, val_loss_early=False)


def try_conf_deep():
    with wandb.init():
        set_seed(SEED)
        k = wandb.config.k
        biased = wandb.config.biased
        lr = wandb.config.lr
        wd = wandb.config.wd
        tr_batch_size = wandb.config.tr_batch_size
        k_filter = wandb.config.k_filter
        genre_threshold = wandb.config.genre_threshold
        balanced = wandb.config.balanced
        init_std = wandb.config.init_std
        dropout = wandb.config.dropout
        n_layers = wandb.config.n_layers
        n_nodes_x_layer = wandb.config.n_nodes_x_layer
        act_func = wandb.config.act_func
        cosine = wandb.config.cosine

        n_users, n_genres, genre_folds = data.get_mr_200k_genre_ratings(SEED, genre_threshold=genre_threshold,
                                                                        k_filter=k_filter, genre_val_size=0.2)
        train_set, val_set = genre_folds.values()
        val_loader = ValDataLoaderMSE(val_set["ratings"], batch_size=256)
        if balanced:
            train_loader = BalancedTrainingDataLoader(train_set["ratings"], tr_batch_size)
        else:
            train_loader = TrainingDataLoader(train_set["ratings"], tr_batch_size)
        mf = DeepMatrixFactorization(train_set["rating_matrix"], n_layers=n_layers, n_nodes_x_layer=n_nodes_x_layer,
                                     act_func=torch.nn.ReLU() if act_func == "relu" else torch.nn.Tanh(), n_factors=k,
                                     biased=biased, cosine=cosine, dropout_p=dropout, init_std=init_std)
        optimizer = Adam(mf.parameters(), lr=lr, weight_decay=wd)
        trainer = MFTrainer(mf, optimizer)
        trainer.train(train_loader, val_loader, None, n_epochs=1000, early=20, verbose=1, wandb_train=True,
                      val_loss_early=True)


def train(config):
    set_seed(SEED)
    n_users, n_genres, genre_folds = data.get_mr_200k_genre_ratings(SEED, binary_ratings=config["binary_ratings"],
                                                                    genre_threshold=config["genre_threshold"],
                                                                    k_filter=config["k_filter"], genre_val_size=0.2)
    train_set, val_set = genre_folds.values()
    val_loader = ValDataLoaderMSE(val_set["ratings"], batch_size=256)
    if config["balanced"] == "pos":
        train_loader = BalancedTrainingDataLoader(train_set["ratings"], config["tr_batch_size"], config["sample_size"])
    elif config["balanced"] == "neg":
        train_loader = BalancedTrainingDataLoaderNeg(train_set["ratings"], config["tr_batch_size"], config["sample_size"])
    else:
        train_loader = TrainingDataLoader(train_set["ratings"], config["tr_batch_size"])
    mf = MatrixFactorization(n_users, n_genres, config["k"], config["biased"], config["init_std"], config["dropout"])
    optimizer = Adam(mf.parameters(), lr=config["lr"], weight_decay=config["wd"])
    trainer = MFTrainer(mf, optimizer)
    trainer.train(train_loader, val_loader, None, n_epochs=1000, early=None, verbose=1,
                  val_loss_early=True, save_path="./model.pth")
    trainer.load_model("./model.pth")
    print(trainer.validate(val_loader, None))


def train_deep(config):
    set_seed(SEED)
    n_users, n_genres, genre_folds = data.get_mr_200k_genre_ratings(SEED, genre_threshold=config["genre_threshold"],
                                                                    k_filter=config["k_filter"], genre_val_size=0.2)
    train_set, val_set = genre_folds.values()
    val_loader = ValDataLoaderMSE(val_set["ratings"], batch_size=256)
    if config["balanced"] == "pos":
        train_loader = BalancedTrainingDataLoader(train_set["ratings"], config["tr_batch_size"], sample_size=0.5)
    elif config["balanced"] == "neg":
        train_loader = BalancedTrainingDataLoaderNeg(train_set["ratings"], config["tr_batch_size"], sample_size=0.5)
    else:
        train_loader = TrainingDataLoader(train_set["ratings"], config["tr_batch_size"])
    mf = DeepMatrixFactorization(train_set["rating_matrix"], n_layers=2, n_nodes_x_layer=100, act_func=torch.nn.ReLU(),
                                 n_factors=50, biased=False, cosine=True, dropout_p=0.5)
    optimizer = Adam(mf.parameters(), lr=config["lr"], weight_decay=config["wd"])
    trainer = MFTrainer(mf, optimizer)
    trainer.train(train_loader, val_loader, None, n_epochs=1000, early=20, verbose=1,
                  val_loss_early=True, save_path="./model.pth")
    trainer.load_model("./model.pth")
    print(trainer.validate(val_loader, None))


def train_sig(config):
    with wandb.init():
        set_seed(SEED)
        n_users, n_genres, genre_folds, mapping_200k = data.get_mr_200k_genre_ratings(SEED, binary_ratings=config["binary_ratings"],
                                                                        genre_threshold=config["genre_threshold"],
                                                                        k_filter=None,
                                                                        genre_val_size=config["genre_val_size"],
                                                                        user_level_split=config["user_level_split"])
        # n_users, n_items, n_genres, item_folds, genre_folds, item_genres_matrix, mapping_100k = data.get_mr_genre_ratings(SEED, genre_threshold=config["genre_threshold"],
        #                                                            binary_ratings=config["binary_ratings"],
        #                                                            k_filter=None, test_size=0.2, val_size=0.1,
        #                                                            genre_val_size=config["genre_val_size"])
        train_set, val_set = genre_folds.values()

        prop_pos = np.sum(train_set["ratings"][:, -1] == 1) / len(train_set["ratings"])

        val_loader = ValDataLoaderMSE(val_set["ratings"], batch_size=256)
        if config["balanced"] == "pos":
            train_loader = BalancedTrainingDataLoader(train_set["ratings"], config["tr_batch_size"],
                                                      sample_size=config["sample_size"])
        elif config["balanced"] == "neg":
            train_loader = BalancedTrainingDataLoaderNeg(train_set["ratings"], config["tr_batch_size"],
                                                         sample_size=config["sample_size"])
        else:
            train_loader = TrainingDataLoader(train_set["ratings"], config["tr_batch_size"])
        mf = MatrixFactorization(n_users, n_genres, config["k"], config["biased"], config["init_std"], config["dropout"],
                                 normalize=True)
        optimizer = Adam(mf.parameters(), lr=config["lr"], weight_decay=config["wd"])
        trainer = MFTrainer(mf, optimizer, True, FocalLoss(alpha=(1 - prop_pos if config["alpha"] is None else config["alpha"]),
                                                     gamma=config["gamma"], reduction=config["reduction"]),
                            config["threshold"],
                            config["beta"])
        trainer.train(train_loader, val_loader, "f-beta", n_epochs=1000, early=config["early"],
                      verbose=1, val_loss_early=False, save_path="./model.pth")
        trainer.load_model("./model.pth")


if __name__ == "__main__":
    configuration = {
        'method': "bayes",
        'metric': {'goal': 'minimize', 'name': 'val_loss'},
        'parameters': {
            'k': {"values": [1, 2, 5, 10, 20, 50, 100, 200, 500]},
            'lr': {"distribution": "log_uniform_values", "min": 0.00001, "max": 0.001},
            'wd': {"distribution": "log_uniform_values", "min": 0.001, "max": 1.},
            'biased': {"values": [1, 0]},
            'tr_batch_size': {"values": [16, 32, 64, 128, 256, 512]},
            'genre_threshold': {'value': None},
            'k_filter': {'value': 5},
            'balanced': {'values': [0, 1]},
            'init_std': {"distribution": "log_uniform_values", "min": 0.0001, "max": 0.01},
            'dropout': {'values': [0., 0.2, 0.5]}}
    }

    configuration_deep = {
        'method': "bayes",
        'metric': {'goal': 'minimize', 'name': 'val_loss'},
        'parameters': {
            'k': {"values": [1, 2, 5, 10, 20, 50, 100, 200, 500]},
            'lr': {"distribution": "log_uniform_values", "min": 0.00001, "max": 0.001},
            'wd': {"distribution": "log_uniform_values", "min": 0.001, "max": 1.},
            'biased': {"value": 0},
            'cosine': {"value": 1},
            'tr_batch_size': {"values": [16, 32, 64, 128, 256, 512]},
            'genre_threshold': {'value': None},
            'k_filter': {'value': 5},
            'balanced': {'values': [0, 1]},
            'init_std': {"distribution": "log_uniform_values", "min": 0.0001, "max": 0.01},
            'dropout': {'values': [0., 0.2, 0.5]},
            'n_layers': {'values': [1, 2, 3]},
            'n_nodes_x_layer': {"values": [20, 50, 100, 200]},
            'act_func': {"values": ["relu", "tanh"]}}
    }

    configuration_sig = {
        'method': "bayes",
        'metric': {'goal': 'maximize', 'name': 'f-beta'},
        'parameters': {
            'k': {"values": [1, 2, 5, 10, 20, 50, 100, 200, 500]},
            'lr': {"distribution": "log_uniform_values", "min": 0.00001, "max": 0.001},
            'wd': {"distribution": "log_uniform_values", "min": 0.001, "max": 1.},
            'biased': {"values": [0, 1]},
            'tr_batch_size': {"values": [16, 32, 64, 128, 256, 512]},
            'init_std': {"distribution": "log_uniform_values", "min": 0.0001, "max": 0.01},
            'dropout': {'values': [0., 0.2, 0.5]},
            "threshold": {'values': [0.3, 0.4, 0.5, 0.6, 0.7]},
            "gamma": {'values': [0, 1, 2, 3, 4]},
            "alpha": {'values': [0.1, 0.3, 0.5, 0.7, 0.9]},
            "reduction": {'values': ["sum", "mean"]}
            }
    }

    sweep_id = wandb.sweep(sweep=configuration_sig, project="likes_genre")
    wandb.agent(sweep_id, function=try_conf_sig, count=300)

    # sweep_id = wandb.sweep(sweep=configuration_deep, project="deep_mf")
    # wandb.agent(sweep_id, function=try_conf_deep, count=100)

    # sweep_id = wandb.sweep(sweep=configuration, project="k_filter_5")
    # wandb.agent(sweep_id, function=try_conf, count=100)

    # sweep_id = wandb.sweep(sweep=configuration, project="validation_50")
    # wandb.agent(sweep_id, function=try_conf, count=100)

    # train({"k": 10, "lr": 0.0001, "wd": 0.001, "biased": 1, "tr_batch_size": 128, "balanced": None, "init_std": 0.001,
    #        "dropout": 0, "genre_threshold": None, "k_filter": 5, "binary_ratings": 0, "sample_size": 1.0})

    # train LikesGenre

    train_sig({"k": 10, "lr": 0.0001, "wd": 0.001, "biased": 1, "tr_batch_size": 128, "balanced": None,
               "init_std": 0.001, "dropout": 0, "genre_threshold": None, "k_filter": 5, "binary_ratings": 1,
               "sample_size": 1.0, "genre_val_size": 0.2, "user_level_split": 0, "threshold": 0.5, "beta": 0.5,
               "gamma": 0, "alpha": 0.5, "early": None, "reduction": "sum"})

    # todo e' importante riflettere lo sbilanciamento anche sul validation set durante la cross validation
    # todo in scikit learn viene prima la class con un ascii precedente
    # todo devo aumentare il numero di true negatives e diminuire il numero di false negative.
    # todo devo aumentare il numero di true negatives per applicare il piu' possibile la regola e diminuire il numero di
    #  false negatives per essere sicuro che non applico la regola su un positivo. Questo significa aumentare la precision
    # todo il pochi ma buoni l'ho ottenuto

    # train_deep({"k": 10, "lr": 0.001, "wd": 0.002, "biased": 1, "tr_batch_size": 512, "balanced": None, "init_std": 0.001,
    #             "dropout": 0, "genre_threshold": None, "k_filter": 5})
    # todo provare a lanciare con un validation piu' ampio tipo 50-50

# todo verificato che sono uno un sottoinsieme dell'altro e i generi sono rimasti 164 in totale
# todo fare esperimenti con il dataset nuovo e vedere se riesco a raggiungere un mse migliore
# todo poi, tenere solamente gli embedding dei utenti e generi del piccolo e usarli per fare transfer learning
# todo provare la deep matrix factorization e vedere come va
# todo provare la cross entropy loss o qualcosa di simile, perche' alla fine ho 1 e -1, quindi non sarebbe neanche
#  regressione il mio task
# todo voglio calcolare l'RMSE sia per positivi che per negativi, per avere un'idea di come il mio algoritmo
#  va in questi casi
# todo non so se sui generi abbia senso dire che se ne peschi uno a caso con cui non ha interagito, allora e' piu'
#  probabile che non gli piaccia, con i film sicuramente questa cosa e' vera
# todo un'altra cosa che posso provare a fare e' undersampling
# todo In general, sensitivity is more important than specificity when the objective is to maximize the number of positive examples that are correctly classified. However, specificity is more important than sensitivity when the objective is to minimize the number of negative examples that are incorrectly classified.
