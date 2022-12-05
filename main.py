from ltnrec.utils import set_seed, train_standard_mf, train_ltn_mf, train_ltn_mf_genres
from ltnrec.data import DataManager

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
# todo basta generare quel file (exp1.json) per i 30 o piu' seed diversi, poi si leggono i vari files e si fa la media
# todo scrivere il codice per fare le grid search e trovare i migliori iperparametri
# todo scrivere il codice per implementare il discorso del cold start, bisogna togliere rating in maniera intelligente.
#  Bisogna togliere piu' ratings per ogni utente
# todo creare una funzione da chiamare per fare il training di uno specifico modello, potrebbe essere una classe
#  che contiene metodi per fare training di tutti i modelli della repo

# set seed
seed = 2022
# load dataset class
data_manager = DataManager("./datasets")
# get dataset that is the union of ml and mr
movie_ratings, movie_genres_matrix, genre_ratings, idx_mapping = data_manager.get_ml_union_mr_dataset()
# get dataset that is the union of movie ratings of ml and genre ratings of mr
g_movie_ratings, g_movie_genres_matrix, g_genre_ratings = data_manager.get_ml_union_mr_genre_ratings_dataset()
exp_name = "exp2"
for mode in ["ml", "ml\mr", "ml&mr"]:  # "ml", "ml \ mr", "ml & mr"
    set_seed(seed)
    # get ml-100k dataset
    ml_dataset = data_manager.get_ml100k_folds(seed, mode=mode, n_neg=200)
    print("------------ Evaluation on %s ------------" % mode)
    print("------------ Training on ml -------------")
    train_standard_mf(ml_dataset, ["hit@10", "ndcg@10"], seed, "%s-%s-%s" % (mode, "ml", "standard_mf"), exp_name)
    train_ltn_mf(ml_dataset, ["hit@10", "ndcg@10"], seed, "%s-%s-%s" % (mode, "ml", "ltn_mf"), exp_name)
    # get fusion folds based on ml-100k folds
    ml_union_mr_dataset_movies = data_manager.get_fusion_folds_given_ml_folds(movie_ratings, ml_dataset.val,
                                                                              ml_dataset.test, idx_mapping)
    # get fusion folds taking also the genre ratings into account
    ml_union_mr_dataset_movies_genres = data_manager.get_fusion_folds_given_ml_folds(movie_ratings, ml_dataset.val,
                                                                                     ml_dataset.test, idx_mapping,
                                                                                     genre_ratings)
    ml_union_mr_dataset_movies_genres.set_item_genres_matrix(movie_genres_matrix)
    # get fusion folds taking only the genre ratings into account
    ml_movies_union_mr_genres_dataset = data_manager.get_fusion_folds_given_ml_folds(g_movie_ratings, ml_dataset.val,
                                                                                     ml_dataset.test,
                                                                                     genre_ratings=g_genre_ratings)
    ml_movies_union_mr_genres_dataset.set_item_genres_matrix(g_movie_genres_matrix)
    print("------------- Training on ml | mr (movies) --------------")
    train_standard_mf(ml_union_mr_dataset_movies, ["hit@10", "ndcg@10"], seed,
                      "%s-%s-%s" % (mode, "ml|mr(movies)", "standard_mf"), exp_name)
    train_ltn_mf(ml_union_mr_dataset_movies, ["hit@10", "ndcg@10"], seed,
                 "%s-%s-%s" % (mode, "ml|mr(movies)", "ltn_mf"), exp_name)
    print("------------- Training on ml | mr (movies + genres) --------------")
    train_standard_mf(ml_union_mr_dataset_movies_genres, ["hit@10", "ndcg@10"], seed,
                      "%s-%s-%s" % (mode, "ml|mr(movies+genres)", "standard_mf"), exp_name)
    train_ltn_mf(ml_union_mr_dataset_movies_genres, ["hit@10", "ndcg@10"], seed,
                 "%s-%s-%s" % (mode, "ml|mr(movies+genres)", "ltn_mf"),
                 exp_name)
    train_ltn_mf_genres(ml_union_mr_dataset_movies_genres, ["hit@10", "ndcg@10"], seed,
                        "%s-%s-%s" % (mode, "ml|mr(movies+genres)",
                                      "ltn_mf_genres"), exp_name)
    print("------------- Training on ml (movies) | mr (genres) --------------")
    # todo controllare se qui vengon risultati sballati perche' ho usato un validation e test diversi da ml val e test
    train_standard_mf(ml_movies_union_mr_genres_dataset, ["hit@10", "ndcg@10"], seed,
                      "%s-%s-%s" % (mode, "ml(movies)|mr(genres)", "standard_mf"), exp_name)
    train_ltn_mf(ml_movies_union_mr_genres_dataset, ["hit@10", "ndcg@10"], seed,
                 "%s-%s-%s" % (mode, "ml(movies)|mr(genres)", "ltn_mf"), exp_name)
    train_ltn_mf_genres(ml_movies_union_mr_genres_dataset, ["hit@10", "ndcg@10"], seed,
                        "%s-%s-%s" % (mode, "ml(movies)|mr(genres)",
                                      "ltn_mf_genres"), exp_name)
