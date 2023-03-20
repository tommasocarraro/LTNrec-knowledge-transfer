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

data = DataManager("./datasets")
SEED = 0


def learn_and_load_likes_genre(config):
    # set seed for reproducibility of the experiment
    set_seed(SEED)
    # get MindReader-200k user-genre ratings
    n_users, n_genres, genre_folds, mapping_200k = data.get_mr_200k_genre_ratings(SEED, binary_ratings=True,
                                                                                  genre_threshold=None,
                                                                                  k_filter=None,
                                                                                  genre_val_size=0.2,
                                                                                  user_level_split=False)
    # create train and validation splits
    train_set, val_set = genre_folds.values()
    # create validation loader
    val_loader = ValDataLoaderMSE(val_set["ratings"], batch_size=256)
    # create training data loader
    train_loader = TrainingDataLoader(train_set["ratings"], config["tr_batch_size"])
    # create Matrix Factorization model
    mf = MatrixFactorization(n_users, n_genres, config["k"], config["biased"], config["init_std"], config["dropout"],
                             normalize=True)
    # create optimizer for learning
    optimizer = Adam(mf.parameters(), lr=config["lr"], weight_decay=config["wd"])
    # create trainer for training the model
    trainer = MFTrainer(mf, optimizer, wandb_train=False,
                        loss=FocalLoss(alpha=config["alpha"], gamma=config["gamma"], reduction=config["reduction"]),
                        threshold=config["threshold"],
                        beta=0.5)
    # train the model using f-best as validation metric since the dataset is imbalanced, early stopping with patience
    # of 30 epochs is used
    trainer.train(train_loader, val_loader, "f-beta", n_epochs=1000, early=30, verbose=1, val_loss_early=False,
                  save_path="./model.pth")
    # load best weights
    trainer.load_model("./model.pth")
    # return predicted user-genre matrix, a 1 means the user likes the genre, a 0 the user does not like the genre
    # if cut-off is False, the continuous output of the model is returned instead
    u_g_matrix = torch.matmul(mf.u_emb.weight.data, torch.t(mf.i_emb.weight.data))
    if mf.biased:
        u_g_matrix = torch.add(u_g_matrix, mf.u_bias.weight.data)
        u_g_matrix = torch.add(u_g_matrix, torch.t(mf.i_bias.weight.data))
    # apply sigmoid to prediction
    u_g_matrix = torch.nn.Sigmoid()(u_g_matrix)
    if config["cut-off"]:
        u_g_matrix = (u_g_matrix >= config["threshold"]).float()
    return u_g_matrix, mapping_200k


if __name__ == "__main__":
    # step 1: learn the LikesGenre predicate and get the user-genre mapping
    config = {"k": 200, "lr": 0.001, "wd": 0.001, "biased": 1, "tr_batch_size": 512, "init_std": 0.001, "dropout": 0,
              "threshold": 0.5, "gamma": 1, "alpha": 0.3, "reduction": "sum", "cut-off": 1}
    likes_genre_matrix, user_genre_mapping = learn_and_load_likes_genre(config)
    # step 2: learn the LikesMovie predicate with LTN without the rule
    # todo qui voglio metterlo come un problema di classificazione
    # todo vedere sbilanciamento dei rating 1, -1 anche per i film, capire cosa voglio minimizzare
    # todo conviene fare direttamente sull'intero dataset e vedere come va, quello con 200k ratings
    # step 3: learn the LikesMovie predicate with LTN with the new rule containing the LikesGenre predicate
    # todo devo implementare la AUC e anche fare un pre-processing del dataset per avere questa AUC e anche un
    #  dataloader per la AUC
    # todo pensare a come fare upsampling della negativa prendendo dati da un altro dataset, pero' non ha
    #  troppissimo senso
