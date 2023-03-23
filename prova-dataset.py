import torch.nn

from ltnrec.data import DataManager
from ltnrec.loaders import ValDataLoaderRatings, TrainingDataLoader, ValDataLoaderRanking, \
    TrainingDataLoaderBPRCustom, TrainingDataLoaderImplicit, TrainingDataLoaderBPRClassic
from ltnrec.models import MatrixFactorization, MFTrainerClassifier, MFTrainerRegression, MFTrainerBPR
from torch.optim import Adam
from ltnrec.models import FocalLoss
from ltnrec.utils import set_seed


def get_pos_prop(ratings):
    return (ratings[:, -1] == 1).sum() / len(ratings)


d = DataManager("./datasets")
set_seed(0)
n_users, n_genres, n_items, genre_folds, movie_folds, item_genres_matrix = d.get_mr_200k_dataset(seed=0,
                                                                                                 val_mode="rating-prediction",
                                                                                                 movie_val_size=0.1,
                                                                                                 movie_test_size=0.2,
                                                                                                 binary_ratings=False)
g_train_set, g_val_set = genre_folds.values()
entire_u_i_matrix, train_set, train_set_small, val_set, test_set = movie_folds.values()

# val_loader = ValDataLoaderRanking(val_set, batch_size=256)
val_loader = ValDataLoaderRatings(val_set["ratings"], batch_size=256)
train_loader = TrainingDataLoader(train_set_small["ratings"], 512)
# train_loader = TrainingDataLoaderImplicit(train_set_small["ratings"], entire_u_i_matrix, 256)
# train_loader = TrainingDataLoaderBPRCustom(train_set_small["ratings"], entire_u_i_matrix, 512)
# train_loader = TrainingDataLoaderBPRClassic(train_set_small["ratings"], entire_u_i_matrix, 256)

mf = MatrixFactorization(n_users, n_items, 200, 1, 0.001, 0)
optimizer = Adam(mf.parameters(), lr=0.001, weight_decay=0.001)
# trainer = MFTrainerClassifier(mf, optimizer,
#                               FocalLoss(alpha=get_pos_prop(train_set_small["ratings"]), gamma=2, reduction="sum"),
#                               False, threshold=0.5)
# trainer = MFTrainerClassifier(mf, optimizer, torch.nn.BCELoss(), False, threshold=0.5)
trainer = MFTrainerRegression(mf, optimizer, torch.nn.MSELoss(), False)
# trainer = MFTrainerBPR(mf, optimizer, False)
trainer.train(train_loader, val_loader, "mse", n_epochs=1000, early=10, verbose=1)

# todo implementare il metodo test per le test metrics
# todo confrontare focal loss con LTN con p diversi per le due regole a seconda dello sbilanciamento
# todo sembrerebbe che la ndcg non sia una buona metrica per vedere se il modello va bene, perche' funziona bene
#  con la BCE loss. Tutti i modelli che abbiamo provato non funzionano bene con la BCE e danno risultati pessimi
# todo molti usano la xavier normal per inizializzare i pesi
# todo modello con stesso seed, stessa loss, metrica diversa, va bene su una metrica e male sulle altre, il
#  problema e' la metrica
# todo implementare l'approccio LTN di classificazione, con un peso diverso per le due classi -> spero funzioni bene
# todo non riesco a trovare un modo per implementare il tutto senza LTN? Magari funziona benissimo
# todo il metodo BPR puo' essere inglobato nella MFRegression? Forse no perche' e' proprio diverso tutto


# Neg prec 0.569 - Pos prec 0.785 - Neg rec 0.639 - Pos rec 0.732 - Neg f 0.602 - Pos f 0.757
# tn 1774 - fp 1004 - fn 1344 - tp 3665
# sensitivity 0.732 - specificity 0.639
