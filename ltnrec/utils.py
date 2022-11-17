import torch
import numpy as np
import random
from ltnrec.models import MatrixFactorization, MFTrainer, LTNTrainerMF, LTNTrainerMFGenres
from ltnrec.loaders import TrainingDataLoader, ValDataLoader, TrainingDataLoaderLTN, TrainingDataLoaderLTNGenres
from torch.optim import Adam


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_standard_mf(n_users, n_items, k, biased, tr, val, test, tr_batch_size, val_batch_size, lr, wd, seed):
    set_seed(seed)
    mf_model = MatrixFactorization(n_users, n_items, k, biased)
    tr_loader = TrainingDataLoader(tr, tr_batch_size)
    val_loader = ValDataLoader(val, val_batch_size)
    test_loader = ValDataLoader(test, val_batch_size)
    trainer = MFTrainer(mf_model, Adam(mf_model.parameters(), lr=lr, weight_decay=wd))
    trainer.train(tr_loader, val_loader, "hit@10", n_epochs=100, early=10, verbose=1, save_path="saved_models/mf.pth")
    trainer.load_model("./saved_models/mf.pth")
    print(trainer.test(test_loader, ["hit@10", "ndcg@10"]))


def train_ltn_mf(n_users, n_items, k, biased, tr, val, test, tr_batch_size, val_batch_size, lr, wd, alpha, seed):
    set_seed(seed)
    mf_model = MatrixFactorization(n_users, n_items, k, biased)
    tr_loader = TrainingDataLoaderLTN(tr, tr_batch_size)
    val_loader = ValDataLoader(val, val_batch_size)
    test_loader = ValDataLoader(test, val_batch_size)
    trainer = LTNTrainerMF(mf_model, Adam(mf_model.parameters(), lr=lr, weight_decay=wd), alpha=alpha)
    trainer.train(tr_loader, val_loader, "hit@10", n_epochs=100, early=10, verbose=1,
                  save_path="saved_models/ltn-mf.pth")
    trainer.load_model("./saved_models/ltn-mf.pth")
    print(trainer.test(test_loader, ["hit@10", "ndcg@10"]))


def train_ltn_mf_genres(n_users, n_items, n_genres, movie_genres, k, biased, tr, val, test, tr_batch_size,
                        val_batch_size, lr, wd, alpha, p, seed):
    set_seed(seed)
    mf_model = MatrixFactorization(n_users, n_items, k, biased)
    # todo sistemare questa cosa della inconsistenza tra indici
    tr_loader = TrainingDataLoaderLTNGenres(tr, n_users, n_items - n_genres, n_genres, tr_batch_size)
    val_loader = ValDataLoader(val, val_batch_size)
    test_loader = ValDataLoader(test, val_batch_size)
    trainer = LTNTrainerMFGenres(mf_model, Adam(mf_model.parameters(), lr=lr, weight_decay=wd), alpha=alpha, p=p,
                                 n_movies=n_items - n_genres, item_genres_matrix=movie_genres)
    trainer.train(tr_loader, val_loader, "hit@10", n_epochs=100, early=10, verbose=1,
                  save_path="saved_models/ltn-mf-genres.pth")
    trainer.load_model("./saved_models/ltn-mf-genres.pth")
    print(trainer.test(test_loader, ["hit@10", "ndcg@10"]))
