import os

import torch.nn.init
import json
from ltnrec.models import MatrixFactorization, LTNTrainerMF, LTNTrainerMFGenres
from ltnrec.loaders import TrainingDataLoaderLTN, ValDataLoader, TrainingDataLoaderLTNGenres
import pickle
from torch.optim import Adam
from ltnrec.utils import set_seed
import numpy as np
from joblib import Parallel, delayed

# get dataset
with open("./final/datasets/ml&mr-ml|mr(movies+genres)-200-1.00-seed_0", 'rb') as dataset_file:
    data = pickle.load(dataset_file)

# set parameter values
# todo provare con degli iperparametri piu' stabili, quindi un valore diverso di k

initializers = [torch.nn.init.uniform_, torch.nn.init.normal_, torch.nn.init.xavier_uniform_,
                torch.nn.init.xavier_normal_, torch.nn.init.kaiming_uniform_, torch.nn.init.kaiming_normal_,
                torch.nn.init.trunc_normal_, torch.nn.init.orthogonal_]

seeds = [i for i in range(0, 10)]


def run_train(seed, initializer):
    k = 20
    biased = 1
    tr_batch_size = 256
    val_batch_size = 512
    lr = 0.0005
    alpha = 0.01
    p = 10
    wd = 0.0001
    set_seed(seed)
    model = MatrixFactorization(data.n_users, data.n_items, k, biased,
                                weight_initializer=initializer)
    tr_loader = TrainingDataLoaderLTN(data.train, tr_batch_size)
    val_loader = ValDataLoader(data.val, val_batch_size)
    test_loader = ValDataLoader(data.test, val_batch_size)
    trainer = LTNTrainerMF(model, Adam(model.parameters(), lr=lr, weight_decay=wd),
                           alpha=alpha, p=p)
    trainer.train(tr_loader, val_loader, "hit@10",
                  n_epochs=100,
                  verbose=1,
                  early=10)

    metric_value = trainer.test(test_loader, ["hit@10"])

    # create file with metric value
    with open("./initialization2/seed_%d-init_%d.json" % (seed, initializers.index(initializer)), "w") as outfile:
        json.dump(metric_value, outfile, indent=4)


def run_train_genres(seed, initializer):
    k = 20
    biased = 1
    tr_batch_size = 512
    val_batch_size = 512
    lr = 0.0001
    alpha = 0.1
    p = 10
    wd = 0.001
    n_sampled_genres = 2
    exists = 0
    set_seed(seed)
    model = MatrixFactorization(data.n_users, data.n_items, k, biased,
                                weight_initializer=initializer)
    tr_loader = TrainingDataLoaderLTNGenres(data.train, data.n_users, data.n_items - data.n_genres, data.n_genres,
                                            n_sampled_genres, tr_batch_size)
    val_loader = ValDataLoader(data.val, val_batch_size)
    test_loader = ValDataLoader(data.test, val_batch_size)
    trainer = LTNTrainerMFGenres(model, Adam(model.parameters(), lr=lr, weight_decay=wd),
                                 alpha=alpha, p=p, n_movies=data.n_items - data.n_genres,
                                 item_genres_matrix=data.item_genres_matrix, exists=exists)
    trainer.train(tr_loader, val_loader, "hit@10",
                  n_epochs=100,
                  verbose=1,
                  early=10)

    metric_value = trainer.test(test_loader, ["hit@10"])

    # create file with metric value
    with open("./initialization2/seed_%d-init_%d.json" % (seed, initializers.index(initializer)), "w") as outfile:
        json.dump(metric_value, outfile, indent=4)


if __name__ == "__main__":
    Parallel(n_jobs=os.cpu_count())(delayed(run_train_genres)(seed, initializer)
                                    for seed in seeds
                                    for initializer in initializers)
    d = {initializers.index(initializer): [] for initializer in initializers}
    for filename in os.listdir("./initialization2"):
        f = os.path.join("./initialization2", filename)
        # checking if it is a file
        if os.path.isfile(f):
            with open(f, 'rb') as json_file:
                metric = json.load(json_file)
            initializer = int(filename.split(".")[0].split("-")[1][-1])
            d[initializer].append(metric["hit@10"])

    d2 = d.copy()
    for initializer in d:
        d2[initializer] = {"mean": np.mean(d[initializer]), "std": np.std(d[initializer])}

    print(d2)



# ltn_model = LTNMFModel()
#
# ltn_model.train_model_wandb("ml&mr-ml|mr(movies+genres)-200-1.00-seed_0", 0, "./final/datasets",
#                             "./final/configs", "./bmxitalia", "final")
