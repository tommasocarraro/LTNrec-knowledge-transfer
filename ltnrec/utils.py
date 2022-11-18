import torch
import numpy as np
import random
from ltnrec.models import MatrixFactorization, MFTrainer, LTNTrainerMF, LTNTrainerMFGenres
from ltnrec.loaders import TrainingDataLoader, ValDataLoader, TrainingDataLoaderLTN, TrainingDataLoaderLTNGenres
from torch.optim import Adam
import json
import os


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def append_to_json(json_name, model_name, result):
    model = model_name.split("-")
    if os.path.exists("./results/%s.json" % (json_name,)):
        # open json file
        with open("./results/%s.json" % (json_name,)) as json_file:
            data = json.load(json_file)
        os.remove("./results/%s.json" % (json_name,))
    else:
        data = {}

    if model[0] not in data:
        data[model[0]] = {model[1]: {model[2]: result}}
    else:
        if model[1] not in data[model[0]]:
            data[model[0]][model[1]] = {model[2]: result}
        else:
            if model[2] not in data[model[0]][model[1]]:
                data[model[0]][model[1]][model[2]] = result

    with open("./results/%s.json" % (json_name,), "w") as outfile:
        json.dump(data, outfile, indent=4)


def train_standard_mf(n_users, n_items, k, biased, tr, val, test, tr_batch_size, val_batch_size, lr, wd, seed,
                      model_name, json_name):
    set_seed(seed)
    mf_model = MatrixFactorization(n_users, n_items, k, biased)
    tr_loader = TrainingDataLoader(tr, tr_batch_size)
    val_loader = ValDataLoader(val, val_batch_size)
    test_loader = ValDataLoader(test, val_batch_size)
    trainer = MFTrainer(mf_model, Adam(mf_model.parameters(), lr=lr, weight_decay=wd))
    trainer.train(tr_loader, val_loader, "hit@10", n_epochs=100, early=10, verbose=1,
                  save_path="./saved_models/%s.pth" % (model_name,))
    trainer.load_model("./saved_models/%s.pth" % (model_name,))
    result = trainer.test(test_loader, ["hit@10", "ndcg@10"])
    append_to_json(json_name, model_name, result)


def train_ltn_mf(n_users, n_items, k, biased, tr, val, test, tr_batch_size, val_batch_size, lr, wd, alpha, seed,
                 model_name, json_name):
    set_seed(seed)
    mf_model = MatrixFactorization(n_users, n_items, k, biased)
    tr_loader = TrainingDataLoaderLTN(tr, tr_batch_size)
    val_loader = ValDataLoader(val, val_batch_size)
    test_loader = ValDataLoader(test, val_batch_size)
    trainer = LTNTrainerMF(mf_model, Adam(mf_model.parameters(), lr=lr, weight_decay=wd), alpha=alpha)
    trainer.train(tr_loader, val_loader, "hit@10", n_epochs=100, early=10, verbose=1,
                  save_path="./saved_models/%s.pth" % (model_name, ))
    trainer.load_model("./saved_models/%s.pth" % (model_name,))
    result = trainer.test(test_loader, ["hit@10", "ndcg@10"])
    append_to_json(json_name, model_name, result)


def train_ltn_mf_genres(n_users, n_items, n_genres, movie_genres, k, biased, tr, val, test, tr_batch_size,
                        val_batch_size, lr, wd, alpha, p, seed, model_name, json_name):
    set_seed(seed)
    mf_model = MatrixFactorization(n_users, n_items, k, biased)
    # todo sistemare questa cosa della inconsistenza tra indici
    tr_loader = TrainingDataLoaderLTNGenres(tr, n_users, n_items - n_genres, n_genres, tr_batch_size)
    val_loader = ValDataLoader(val, val_batch_size)
    test_loader = ValDataLoader(test, val_batch_size)
    trainer = LTNTrainerMFGenres(mf_model, Adam(mf_model.parameters(), lr=lr, weight_decay=wd), alpha=alpha, p=p,
                                 n_movies=n_items - n_genres, item_genres_matrix=movie_genres)
    trainer.train(tr_loader, val_loader, "hit@10", n_epochs=100, early=10, verbose=1,
                  save_path="./saved_models/%s.pth" % (model_name,))
    trainer.load_model("./saved_models/%s.pth" % (model_name,))
    result = trainer.test(test_loader, ["hit@10", "ndcg@10"])
    append_to_json(json_name, model_name, result)
