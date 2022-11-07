import torch
import numpy as np
from ltnrec.data import MovieLensMR
from ltnrec.loaders import ValDataLoader, TrainingDataLoaderLTN
from ltnrec.models import MatrixFactorization, LTNTrainerMF
from torch.optim import Adam

# todo vedere se su MindReader ho solo i generi, mi serve pero' la regola dei generi, senza quella la informazione
#  non si propaga perche' gli utenti sono disgiunti
# todo con film + generi l'informazione si progaga perche' l'embedding per uno stesso utente di MindReader cambia sia
#  se gli piace un genere che se gli piace un film, quindi prende entrambe le informazioni, e lo stesso il film e il
#  genere cambiano insieme all'utente. Poi, se il film e' stato visto da altri utenti, e' come se l'informazione di
#  questo utente venisse trasmessa agli altri
# todo solo con i generi questa cosa non funziona perche' gli utenti non condividono film. Potrebbe essere che aiuti lo
#  stesso?
# todo il test set ha sia film che sono nell'intersezione che film che sono solo in movielens, sarebbe da fare anche
#  un test set che ha solo film in movielens, in questa maniera si rende piu' difficile la propagazione
#  dell'informazione senza l'utilizzo della regola con LikesGenre (seconda regola) - fare anche un test set solo di
#  intersezione
# todo se uso solo i generi su mindreader, cosa succede quando passo a LikesGenre l'embedding in un utente che e' di ML?

# load dataset class
data = MovieLensMR("./datasets")
# get fusion dataset
ratings, _, _, genre_ratings, ml_to_new_idx, _, user_mapping_ml, _ = data.create_ml_mr_fusion()
# begin experiment
for mode in ["only_ml"]:
    # get ml-100k dataset
    ml_tr, ml_val, ml_test, ml_n_users, ml_n_items = data.get_ml100k_folds(123, mode=mode)
    # get fusion folds based on ml-100k folds
    fusion_train, fusion_val, fusion_test, fusion_n_users, fusion_n_items = \
        data.get_fusion_folds(ratings, ml_to_new_idx, user_mapping_ml, ml_val, ml_test)
    # get fusion folds taking also the genre ratings into account
    fusion_genres_train, _, _, fusion_genres_n_users, fusion_genres_n_items = \
        data.get_fusion_folds(ratings, ml_to_new_idx, user_mapping_ml, ml_val, ml_test, genre_ratings)
    # create validation/test loaders
    val_loader_ml = ValDataLoader(ml_val, 512)
    val_loader_fusion = ValDataLoader(fusion_val, 512)
    test_loader_ml = ValDataLoader(ml_test, 512)
    test_loader_fusion = ValDataLoader(fusion_test, 512)
    # create train loaders
    tr_loader_ml = TrainingDataLoaderLTN(ml_tr, 256)
    tr_loader_fusion = TrainingDataLoaderLTN(fusion_train, 256)
    tr_loader_fusion_genres = TrainingDataLoaderLTN(fusion_genres_train, 256)
    # create MF models
    torch.manual_seed(123)
    np.random.seed(123)
    mf_model_ml = MatrixFactorization(ml_n_users, ml_n_items, 1, biased=True)
    mf_model_fusion = MatrixFactorization(fusion_n_users, fusion_n_items, 1, biased=True)
    mf_model_fusion_genres = MatrixFactorization(fusion_genres_n_users, fusion_genres_n_items, 1, biased=True)
    # create model trainers
    trainer_ml = LTNTrainerMF(mf_model_ml, Adam(mf_model_ml.parameters(), lr=0.001, weight_decay=0.0001), 0.05)
    trainer_fusion = LTNTrainerMF(mf_model_fusion, Adam(mf_model_fusion.parameters(), lr=0.001, weight_decay=0.0001), 0.05)
    trainer_fusion_genres = LTNTrainerMF(mf_model_fusion_genres, Adam(mf_model_fusion_genres.parameters(), lr=0.001,
                                                                      weight_decay=0.0001), 0.05)
    # train models
    trainer_ml.train(tr_loader_ml, val_loader_ml, "hit@10", n_epochs=50, early=10, verbose=1,
                     save_path="./models/ml.pth")
    trainer_fusion.train(tr_loader_fusion, val_loader_fusion, "hit@10", n_epochs=50, early=10, verbose=1,
                         save_path="./models/fusion.pth")
    trainer_fusion_genres.train(tr_loader_fusion_genres, val_loader_fusion, "hit@10", n_epochs=50, early=10, verbose=1,
                                save_path="./models/fusion_genres.pth")
    # load best models
    trainer_ml.load_model("./models/ml.pth")
    trainer_fusion.load_model("./models/fusion.pth")
    trainer_fusion_genres.load_model("./models/fusion_genres.pth")
    # test models
    print("Experiment with mode: %s" % (mode if mode is not None else "random"))
    print("LTN trained on ml-100k")
    print(trainer_ml.test(test_loader_ml, ["hit@10", "ndcg@10"]))
    print("LTN trained on the fusion of ml-100k and MindReader (only movies)")
    print(trainer_fusion.test(test_loader_fusion, ["hit@10", "ndcg@10"]))
    print("LTN trained on the fusion of ml-100k and MindReader (with genres in the MF)")
    print(trainer_fusion_genres.test(test_loader_fusion, ["hit@10", "ndcg@10"]))
