from ltnrec.utils import set_seed, train_standard_mf, train_ltn_mf, train_ltn_mf_genres
from ltnrec.data import MovieLensMR

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

# set seed
seed = 123
# load dataset class
data = MovieLensMR("./datasets")
# get fusion dataset
ratings, movie_genres, _, genre_ratings, ml_to_new_idx, _, user_mapping_ml, _ = data.create_ml_mr_fusion()
# get fusion dataset only genres
g_ratings, g_movie_genres, g_genre_ratings, g_user_mapping, g_item_mapping, _ = data.create_ml_mr_fusion_only_genres()
# begin experiment
for mode in [None, "only_ml", "only_fusion"]:  # None, "only_ml", "only_fusion"
    set_seed(seed)
    # get ml-100k dataset
    ml_tr, ml_val, ml_test, ml_n_users, ml_n_items = data.get_ml100k_folds(seed, mode=mode, n_neg=200)
    print(mode if mode is not None else "random")
    print("Training on ratings of ml-100k")
    train_standard_mf(ml_n_users, ml_n_items, 1, True, ml_tr, ml_val, ml_test, 256, 512, 0.001, 0.0001, seed)
    train_ltn_mf(ml_n_users, ml_n_items, 1, True, ml_tr, ml_val, ml_test, 256, 512, 0.001, 0.0001, 0.05, seed)
    # get fusion folds based on ml-100k folds
    fusion_train, fusion_val, fusion_test, fusion_n_users, fusion_n_items = \
        data.get_fusion_folds(ratings, ml_to_new_idx, user_mapping_ml, ml_val, ml_test)
    # get fusion folds taking also the genre ratings into account
    fusion_genres_train, _, _, fusion_genres_n_users, fusion_genres_n_items = \
        data.get_fusion_folds(ratings, ml_to_new_idx, user_mapping_ml, ml_val, ml_test, genre_ratings)
    # get fusion folds taking only the genre ratings into account
    fusion_g_genres_train, _, _, fusion_g_genres_n_users, fusion_g_genres_n_items = \
        data.get_fusion_folds(g_ratings, g_item_mapping, g_user_mapping, ml_val, ml_test, g_genre_ratings)
    print("Training on entire dataset without ratings on genres")
    train_standard_mf(fusion_n_users, fusion_n_items, 1, True, fusion_train, fusion_val, fusion_test, 256,
                      512, 0.001, 0.0001, seed)
    train_ltn_mf(fusion_n_users, fusion_n_items, 1, True, fusion_train, fusion_val, fusion_test, 256, 512,
                 0.001, 0.0001, 0.05, seed)
    print("Training on entire dataset with ratings on genres")
    train_standard_mf(fusion_genres_n_users, fusion_genres_n_items, 1, True, fusion_genres_train,
                      fusion_val, fusion_test, 256, 512, 0.001, 0.0001, seed)
    train_ltn_mf(fusion_genres_n_users, fusion_genres_n_items, 1, True, fusion_genres_train, fusion_val,
                 fusion_test, 256, 512, 0.001, 0.0001, 0.05, seed)
    train_ltn_mf_genres(fusion_genres_n_users, fusion_genres_n_items, len(data.genres), movie_genres, 1, True,
                        fusion_genres_train, fusion_val, fusion_test, 256, 512, 0.001, 0.0001, 0.05, 2, 2022)
    print("Training on ratings of ml-100k and genre ratings of MindReader")
    train_standard_mf(fusion_g_genres_n_users, fusion_g_genres_n_items, 1, True, fusion_g_genres_train,
                      ml_val, ml_test, 256, 512, 0.001, 0.0001, seed)
    train_ltn_mf(fusion_g_genres_n_users, fusion_g_genres_n_items, 1, True, fusion_g_genres_train, ml_val,
                 ml_test, 256, 512, 0.001, 0.0001, 0.05, seed)
    train_ltn_mf_genres(fusion_g_genres_n_users, fusion_g_genres_n_items, len(data.genres), g_movie_genres, 1, True,
                        fusion_g_genres_train, ml_val, ml_test, 256, 512, 0.001, 0.0001, 0.05, 2, 2022)
