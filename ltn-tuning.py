from ltnrec.data import DataManager
from ltnrec.loaders import ValDataLoaderRatings, TrainingDataLoaderLTNClassification
from ltnrec.models import MatrixFactorization, LTNTrainerMFClassifier
from torch.optim import Adam
from ltnrec.utils import set_seed
import wandb


def try_conf():
    with wandb.init():
        set_seed(0)
        k = wandb.config.k
        biased = wandb.config.biased
        tr_b_size = wandb.config.tr_b_size
        lr = wandb.config.lr
        init_std = wandb.config.init_std
        wd = wandb.config.wd
        p_sat_agg = wandb.config.p_sat_agg

        train_loader = TrainingDataLoaderLTNClassification(train_set_small["ratings"], tr_b_size)
        mf = MatrixFactorization(n_users, n_items, k, biased, init_std, dropout_p=0.0, normalize=True)
        optimizer = Adam(mf.parameters(), lr=lr, weight_decay=wd)
        trainer = LTNTrainerMFClassifier(mf, optimizer, p_pos=0.2, p_neg=0.2, p_sat_agg=p_sat_agg,
                                         wandb_train=True, threshold=0.5)
        trainer.train(train_loader, val_loader, "fbeta-1.0", n_epochs=1000, early=10, verbose=1)


if __name__ == "__main__":
    configuration = {
        'method': "bayes",
        'metric': {'goal': 'maximize', 'name': 'auc'},
        'parameters': {
            'k': {"values": [1, 2, 5, 10, 20, 50, 100, 200]},  # [1, 16, 32, 64, 128, 256, 512]
            # slow learning rate allows to obtain a stable validation metric
            'lr': {"distribution": "log_uniform_values", "min": 0.01, "max": 0.1},
            # high reg penalty allows to compensate for the large latent factor size
            'wd': {"distribution": "log_uniform_values", "min": 0.00001, "max": 0.001},
            'biased': {"values": [1, 0]},
            'tr_b_size': {"values": [16, 32, 64, 128, 256, 512]},
            'init_std': {"distribution": "log_uniform_values", "min": 0.001, "max": 0.1},
            'p_sat_agg': {"values": list(range(2, 5))}
        }
    }
    d = DataManager("./datasets")
    n_users, n_genres, n_items, genre_folds, movie_folds, item_genres_matrix = d.get_mr_200k_dataset(seed=0,
                                                                                                     val_mode="rating-prediction",
                                                                                                     movie_val_size=0.1,
                                                                                                     movie_test_size=0.2)
    g_train_set, g_val_set = genre_folds.values()
    entire_u_i_matrix, train_set, train_set_small, val_set, test_set = movie_folds.values()
    val_loader = ValDataLoaderRatings(val_set["ratings"], 256)

    sweep_id = wandb.sweep(sweep=configuration, project="ltn-tuning")
    wandb.agent(sweep_id, function=try_conf, count=50)

    # todo usare un sampler bilanciato
    # todo controllare bene cosa succede con le metriche, perche' ora ho risultati diversi dall'altro giorno, capire cosa ho cambiato e cosa sta succedendo, soprattuto quella unione di utenti fatta

