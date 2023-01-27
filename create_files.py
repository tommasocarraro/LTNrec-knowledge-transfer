from ltnrec.data import DataManager
from scipy.sparse import csr_matrix
from ltnrec.loaders import ValDataLoaderRandomSplit, TrainingDataLoader
from ltnrec.models import MatrixFactorization, MFTrainer
from torch.optim import Adam
import numpy as np
import wandb

data = DataManager("./datasets")
n_users = data.ml_100_ratings[0].nunique()
n_items = data.ml_100_ratings[1].nunique()
train_set, test_set = DataManager.get_train_test_split(data.ml_100_ratings, test_size=0.2, seed=0)
train_set, val_set = DataManager.get_train_test_split(train_set, test_size=0.1, seed=0)
tr_u_i_matrix = csr_matrix((np.ones(len(train_set)), (list(train_set[0]), list(train_set[1]))),
                           shape=(n_users, n_items))
val_u_i_matrix = csr_matrix((np.ones(len(val_set)), (list(val_set[0]), list(val_set[1]))),
                            shape=(n_users, n_items))

val_loader = ValDataLoaderRandomSplit(tr_u_i_matrix, val_u_i_matrix, batch_size=256)
train = train_set.to_dict("records")
train = np.array([(rating[0], rating[1], rating[2]) for rating in train])


def search_run():
    with wandb.init(project="prova-exact"):
        k = wandb.config.k
        biased = wandb.config.biased
        lr = wandb.config.lr
        wd = wandb.config.wd
        tr_batch_size = wandb.config.tr_batch_size
        train_loader = TrainingDataLoader(train, tr_batch_size)
        mf = MatrixFactorization(n_users, n_items, k, biased)
        trainer = MFTrainer(mf, Adam(mf.parameters(), lr=lr, weight_decay=wd))
        trainer.train(train_loader, val_loader, "ndcg@10", early=10, verbose=1, wandb_train=True)


if __name__ == "__main__":
    wandb.login()
    configuration = {
        'method': "bayes",
        'metric': {'goal': 'maximize', 'name': 'ndcg@10'},
        'parameters': {
            'k': {"values": [1, 8, 16, 32, 64, 128, 256]},
            'lr': {"distribution": "log_uniform_values",
                   "min": 0.00001,
                   "max": 1},
            'wd': {"distribution": "log_uniform_values",
                   "min": 0.00001,
                   "max": 0.1},
            'biased': {"values": [0, 1]},
            'tr_batch_size': {"values": [64, 128, 256, 512]}
        }
    }
    sweep_id = wandb.sweep(sweep=configuration, project="prova-exact")
    wandb.agent(sweep_id, function=search_run, project="prova-exact", count=30)

    # trovare le prestazioni del modello cosi poi fare l'esperimento con il sample e vedere la differenza
