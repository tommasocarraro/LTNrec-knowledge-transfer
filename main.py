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
# todo con la formula avviene effettivo reasoning sui generi e questo aiuta un sacco il modello perche' l'ablation
#  study lo dimostra (abbiamo un esperimento con lo stesso modello con e senza formula)

# set seed
seed = 123
# load dataset class
data = MovieLensMR("./datasets")
# get dataset that is the union of ml and mr
movie_ratings, movie_genres_matrix, _, genre_ratings, ml_to_new_idx, _, user_mapping_ml, _ = \
    data.get_ml_union_mr_dataset()
# get dataset that is the union of movie ratings of ml and genre ratings of mr
g_movie_ratings, g_movie_genres_matrix, g_genre_ratings, g_user_mapping, g_item_mapping, _ = \
    data.get_ml_union_mr_genre_ratings_dataset()
for mode in ["ml", "ml \ mr", "ml & mr"]:
    set_seed(seed)
    # get ml-100k dataset
    ml_tr, ml_val, ml_test, ml_n_users, ml_n_items = data.get_ml100k_folds(seed, mode=mode, n_neg=200)
    print("------------ Evaluation on %s ------------" % mode)
    print("------------ Training on ml -------------")
    # train_standard_mf(ml_n_users, ml_n_items, ml_tr, ml_val, ml_test, ["hit@10", "ndcg@10"], seed,
    #                   "%s-%s-%s" % (mode, "ml", "standard_mf"), "exp1")
    # train_ltn_mf(ml_n_users, ml_n_items, ml_tr, ml_val, ml_test, ["hit@10", "ndcg@10"], seed,
    #              "%s-%s-%s" % (mode, "ml", "ltn_mf"), "exp1")
    # get fusion folds based on ml-100k folds
    fusion_train, fusion_val, fusion_test, fusion_n_users, fusion_n_items = \
        data.get_fusion_folds(movie_ratings, ml_to_new_idx, user_mapping_ml, ml_val, ml_test)
    # get fusion folds taking also the genre ratings into account
    fusion_genres_train, _, _, fusion_genres_n_users, fusion_genres_n_items = \
        data.get_fusion_folds(movie_ratings, ml_to_new_idx, user_mapping_ml, ml_val, ml_test, genre_ratings)
    # get fusion folds taking only the genre ratings into account
    fusion_g_genres_train, _, _, fusion_g_genres_n_users, fusion_g_genres_n_items = \
        data.get_fusion_folds(g_movie_ratings, g_item_mapping, g_user_mapping, ml_val, ml_test, g_genre_ratings)
    # print("------------- Training on ml | mr (movies) --------------")
    # train_standard_mf(fusion_n_users, fusion_n_items, fusion_train,
    #                   fusion_val, fusion_test, ["hit@10", "ndcg@10"], seed,
    #                   "%s-%s-%s" % (mode, "ml | mr (movies)", "standard_mf"), "exp1")
    # train_ltn_mf(fusion_n_users, fusion_n_items, fusion_train, fusion_val, fusion_test, ["hit@10", "ndcg@10"], seed,
    #              "%s-%s-%s" % (mode, "ml | mr (movies)", "ltn_mf"), "exp1")
    # print("------------- Training on ml | mr (movies + genres) --------------")
    # train_standard_mf(fusion_genres_n_users, fusion_genres_n_items, fusion_genres_train,
    #                   fusion_val, fusion_test, ["hit@10", "ndcg@10"], seed,
    #                   "%s-%s-%s" % (mode, "ml | mr (movies + genres)", "standard_mf"), "exp1")
    # train_ltn_mf(fusion_genres_n_users, fusion_genres_n_items, fusion_genres_train, fusion_val,
    #              fusion_test, ["hit@10", "ndcg@10"], seed,
    #              "%s-%s-%s" % (mode, "ml | mr (movies + genres)", "ltn_mf"),
    #              "exp1")
    # train_ltn_mf_genres(fusion_genres_n_users, fusion_genres_n_items, len(data.genres), movie_genres_matrix,
    #                     fusion_genres_train, fusion_val, fusion_test, ["hit@10", "ndcg@10"], 2022,
    #                     "%s-%s-%s" % (mode, "ml | mr (movies + genres)",
    #                                   "ltn_mf_genres"), "exp1")
    # print("------------- Training on ml (movies) | mr (genres) --------------")
    # train_standard_mf(fusion_g_genres_n_users, fusion_g_genres_n_items, fusion_g_genres_train,
    #                   ml_val, ml_test, ["hit@10", "ndcg@10"], seed,
    #                   "%s-%s-%s" % (mode, "ml (movies) | mr (genres)", "standard_mf"), "exp1")
    # train_ltn_mf(fusion_g_genres_n_users, fusion_g_genres_n_items, fusion_g_genres_train, ml_val,
    #              ml_test, ["hit@10", "ndcg@10"], seed,
    #              "%s-%s-%s" % (mode, "ml (movies) | mr (genres)", "ltn_mf"), "exp1")
    train_ltn_mf_genres(fusion_g_genres_n_users, fusion_g_genres_n_items, len(data.genres), g_movie_genres_matrix,
                        fusion_g_genres_train, ml_val, ml_test, ["hit@10", "ndcg@10"], 2022,
                        "%s-%s-%s" % (mode, "ml (movies) | mr (genres)",
                                      "ltn_mf_genres"), "exp1")
