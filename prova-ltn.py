from ltnrec.data import DataManager
from ltnrec.loaders import ValDataLoaderRatings, ValDataLoaderRanking, TrainingDataLoaderLTNRegression, \
    TrainingDataLoaderLTNClassificationSampling, TrainingDataLoaderLTNClassification, TrainingDataLoaderLTNBPR
from ltnrec.models import MatrixFactorization, MFTrainerClassifier, MFTrainerRegression, MFTrainerBPR, \
    LTNTrainerMFRegression, LTNTrainerMFClassifier, LTNTrainerMFBPR
from torch.optim import Adam
from ltnrec.models import FocalLoss
from ltnrec.utils import set_seed


def get_pos_prop(ratings):
    return (ratings[:, -1] == 1).sum() / len(ratings)


d = DataManager("./datasets")
set_seed(0)
n_users, n_genres, n_items, genre_folds, movie_folds, item_genres_matrix = d.get_mr_200k_dataset(seed=0,
                                                                                                 val_mode="rating-prediction",
                                                                                                 movie_test_size=0.2,
                                                                                                 movie_val_size=0.1)
g_train_set, g_val_set = genre_folds.values()
entire_u_i_matrix, train_set, train_set_small, val_set, test_set = movie_folds.values()

# val_loader = ValDataLoaderRanking(val_set, batch_size=256)
val_loader = ValDataLoaderRatings(val_set["ratings"], batch_size=256)
# train_loader = TrainingDataLoader(train_set_small["ratings"], 512)
# train_loader = TrainingDataLoaderImplicit(train_set_small["ratings"], entire_u_i_matrix, 256)
# train_loader = TrainingDataLoaderBPR(train_set_small["ratings"], entire_u_i_matrix, 512)
# train_loader = TrainingDataLoaderBPRClassic(train_set_small["ratings"], entire_u_i_matrix, 256)
# train_loader = TrainingDataLoaderLTNRegression(train_set_small["ratings"], 512)
# train_loader = TrainingDataLoaderLTNClassificationSampling(train_set_small["ratings"], 256)
train_loader = TrainingDataLoaderLTNClassification(train_set_small["ratings"], 512)
# train_loader = TrainingDataLoaderLTNBPR(train_set_small["ratings"], entire_u_i_matrix, 512)

mf = MatrixFactorization(n_users, n_items, 200, 1, 0.01, 0, normalize=True)
optimizer = Adam(mf.parameters(), lr=0.01, weight_decay=0.0001)
# todo LTN non funziona bene se i pesi sono inizializzati troppo piccoli e con un learning rate troppo basso e con
#  una regolarizzazione troppo bassa. In questi casi va molto male!!
# trainer = MFTrainerClassifier(mf, optimizer,
#                               FocalLoss(alpha=get_pos_prop(train_set_small["ratings"]), gamma=2, reduction="sum"),
#                               False, threshold=0.5)
# trainer = MFTrainerClassifier(mf, optimizer, torch.nn.BCELoss(), False, threshold=0.5)
# trainer = MFTrainerRegression(mf, optimizer, torch.nn.MSELoss(), False)
# trainer = MFTrainerBPR(mf, optimizer, False)
# todo riflettere sugli scheduling del p
# todo ricordarsi che un valori tra 0 e 1 al quadrato diventa ancora piu' piccolo
# todo implementare un loader bilanciato per LTN, per fare sample di batches correttamente
# todo il vero vincolo di LTN e' quando si hanno problemi sulla loss. Ad esempio una loss basata su sommatoria
#  non si puo' usare
# todo confrontare con focal loss come va, vedere come ottenere risulati simili
# todo sembra quasi impossibile far spostare LTN dal decision boundary sui problemi un minimo piu' complessi di una classificazione binaria toy
# todo come dice Luciano ce ne possiamo fregare della loss finale e non metterla in zero uno e fare tipo la somma dei valori di verita' e cosi via
# trainer = LTNTrainerMFRegression(mf, optimizer, alpha=1, exp=2, p=2, wandb_train=False)
trainer = LTNTrainerMFClassifier(mf, optimizer, p_pos=2, p_neg=2, p_sat_agg=2, wandb_train=False, threshold=0.5)
# trainer = LTNTrainerMFBPR(mf, optimizer, alpha=2, p=2, wandb_train=False)
trainer.train(train_loader, val_loader, "fbeta-1.0", n_epochs=1000, early=10, verbose=1)
