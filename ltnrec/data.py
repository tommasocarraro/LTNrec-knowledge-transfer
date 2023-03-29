import pandas as pd
import os
import numpy as np
from fuzzywuzzy import fuzz
import random
from SPARQLWrapper import SPARQLWrapper, JSON, POST
from scipy.sparse import csr_matrix
from ltnrec.utils import set_seed
from sklearn.model_selection import train_test_split as train_test_split_sklearn
import torch


# todo ci sono 457 film matchati tra i due dataset per i quali pero' non ci sono ratings su mindreader, quindi e' un
#  match inutile perche' non abbiamo un test set e nemmeno una ground truth per quei film
# todo bisognerebbe discutere se possano essere comunque utili con la scusa dei generi, magari quelli propogano
#  informazione, pero' non avendo quei film dei rating, l'informazione si propaga solo attraverso il LikesGenre appreso
# todo discutere di questa cosa con Luciano e Alessandro
# todo moltissimi film su mindreader non hanno ratings o hanno solo ratings unknown, cosa posso fare? ->
#  possono essere usati come cold start in casi futuri, pero' non ho rating su quelli e quindi non posso fare dei
#  test set -> per ora li ho eliminati tutti
# todo il problema e' che molti di questi facevano parte dell'intersezione tra i due dataset
# todo osservazione interessante: alcuni film non hanno ratings, potrebbero essere trattati come dei casi di
#  cold-start in futuro, e' che non li puoi valutare
# todo per simulare il cold-start si possono prendere tutti i rating di un film e metterli in test, in modo tale che
#  durante il training non ci siano piu' rating per quel film, magari non tutti ma una grande percentuale, soprattutto
#  per gli utenti con piu' ratings
# todo la regola dei generi la forziamo solo per quelle triple in cui mi manca il Likes a destra, devo pescare dalle
#  triple mancanti nel dataset e utilizzare quella regola. Forse bisogna anche fare in modo che gli item siano nello
#  stesso batch? Non e' detto, e' da pensare
# todo DUBBIO: se ci basiamo su un modello predetto di Likes per fare la seconda formula, secondo me potrebbe
#  corrompere tutto. Non e' piu' una knowledge al 100% affidabile -> potrebbe fare casino e' quello che vogliamo
#  verificare se la knowledge viene trasferita oppure no
# todo rendere completi gli esperimenti aggiungendo anche le altre casistiche di dataset
# todo anche se abbiamo creato i set di val e test cosi, e' comunque fuori distribuzione, perhce' il training ha una
#  distribuzione diversa. Non capisco perche' non debba funzionare come avevo deciso di fare le cose io. Era comunque
#  allenato su quegli utenti e item
# todo e' corretto fare questa cosa? secondo me no, pero' senza farla e' ovvio che funziona peggio
#  perche' probabilmente e' un fold su cui va male, restringersi solamente a movielens aiuta ad
#  eliminare i film su cui ha molti dati e che potrebbero essere quindi anche i piu' popolari
# todo La regola dei generi può fare anche ground truth correction. Non gli piace un genere ma su movielens gli piace
#  un film con quel genere? Correggo
# todo Forse dovrei forzare solo per i film su movielens la regola? Non penso cambi tanto forzarla dapertutto, anzi
#  forse e' meglio, cosi inferiamo ancora meglio i buchi della matrice
# todo bisognerebbe provare con la BPR loss
# todo l'interfaccia del framework permette di utilizzare un qualsiasi metodo che utilizza degli indici, basta creare
#  un metodo che si interfaccia in questo modo
# todo creare il predicato hasGenre, che dati gli indici dei generi, li converte tra 0 e 18 e poi fornisce i valori
#  richiesti
# todo forse ha senso che se a uno piace comedy horror, almeno mettiamo che gli piace il genere piu' alto nella
#  gerarchia?

class Dataset:
    def __init__(self, train, val, test, n_users, n_items, name):
        self.train = train
        self.val = val
        self.test = test
        self.n_users = n_users
        self.n_items = n_items
        self.name = name


class DatasetWithGenres(Dataset):
    def __init__(self, train, val, test, n_users, n_items, name, n_genres, n_genre_ratings, item_genres_matrix=None):
        super(DatasetWithGenres, self).__init__(train, val, test, n_users, n_items, name)
        self.n_genres = n_genres
        self.n_genre_ratings = n_genre_ratings
        self.item_genres_matrix = item_genres_matrix

    def set_item_genres_matrix(self, item_genres_matrix):
        self.item_genres_matrix = item_genres_matrix


def send_query(query):
    """
    It sends a sparql query to the Wikidata ontology and returns the result.
    """
    endpoint_url = 'https://query.wikidata.org/sparql'
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 12_6) AppleWebKit/537.36 (KHTML, like Gecko) ' \
                 'Chrome/106.0.0.0 Safari/537.36'
    sparql = SPARQLWrapper(endpoint_url)
    sparql.addCustomHttpHeader('User-Agent', user_agent)
    sparql.addCustomHttpHeader('Retry-After', '2')
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setMethod(POST)
    result = sparql.query().convert()['results']['bindings']
    return result


class DataManager:
    """
    Class that manages the dataset. The dataset is a fusion of the MovieLens-100k dataset with the MindReader dataset.

    The dataset is created by finding a match between the movies of ml-100k and the movies of MindReader. After the
    match is found, the movies in the joint set (matched movies) will share the same index in the final dataset.
    Instead, for the movies that do not belong to the joint set a new index will be created, both for movies in
    ml-100k and MindReader.
    The users of the two datasets are disjoint. For these users a new indexing is created in such a way they live in the
    same index space.
    MindReader has also ratings for movie genres and other entities. The ratings on genres are added to the final
    datasets.
    MindReader specifies over 150 movie genres. To simplify the learning procedure, we kept only the movie genres which
    are also on the ml-100k dataset. These are the most popular genres. The other genres were subsets of these genres.

    The idea behind the creation of this dataset is to test whether the addition of ratings from other users and other
    movies can help in reaching better results on the Top-N recommendation task. In particular, we are interested
    in transferring knowledge from MindReader to ml-100k. This can be done with specific axioms implemented in Logic
    Tensor Networks. The main idea is to learn a model to model genre preferences on the users of MindReader and then
    use this model to infer such ratings for the ml-100k dataset. This model should help in obtaining better results
    thanks to the use of a specific axiom in LTN.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.check()
        # these are the 18 genres of ml-100k - we keep these genres also for MindReader
        self.genres = ("Action Film", "Adventure Film", "Animated Film", "Children's Film", "Comedy Film", "Crime Film",
                       "Documentary Film", "Drama Film", "Fantasy Film", "Film Noir", "Horror Film", "Musical Film",
                       "Mystery Film", "Romance Film", "Science Fiction Film", "Thriller Film", "War Film",
                       "Western Film")
        # process ml-100k
        if not os.path.exists(os.path.join(self.data_path, "ml-100k/processed/item_mapping.csv")):
            self.process_ml_100k()
        # process MindReader
        if not os.path.exists(os.path.join(self.data_path, "mindreader/processed/item_mapping.csv")):
            self.process_mr()
        # save the files which are mostly used by the pre-processing procedure
        self.ml_100_movie_info = pd.read_csv(os.path.join(self.data_path, "ml-100k/processed/u.item"),
                                             encoding="iso-8859-1", header=None).to_dict("records")
        self.ml_100_ratings = pd.read_csv(os.path.join(self.data_path, "ml-100k/processed/u.data"),
                                          sep="\t", header=None).to_dict("records")
        self.mr_movie_info = pd.read_csv(os.path.join(self.data_path,
                                                      "mindreader/processed/movies.csv")).to_dict("records")
        self.mr_movie_ratings = pd.read_csv(os.path.join(self.data_path,
                                                         "mindreader/processed/movie_ratings.csv")).to_dict("records")
        self.mr_genre_ratings = pd.read_csv(os.path.join(self.data_path,
                                                         "mindreader/processed/genre_ratings.csv")).to_dict("records")
        # create the match between the movies of ml-100k and the movies of MindReader
        self.ml_to_mr = self.create_mapping()

    def check(self):
        """
        Check the presence of all needed files:
        1. entities: contains the entities (movies, genres, actors, ...) of the MindReader KG;
        2. ratings: contains the user-entity ratings of the MindReader dataset;
        3. triples: contains the relationships between the entities of the MindReader KG. For example, who is the actor
        of the movie? Which is the genre? Which is the publication year? All this information is included in the 
        triples file;
        4. u.data (MovieLens-100k): contains the user-item ratings of the MovieLens-100k dataset;
        5. u.item (MovieLens-100k): contains the content information of the movies in MovieLens-100k;
        6. movies (MovieLens-latest): contains the content information of the movies in the MovieLens latest dataset. 
        This dataset is used because it contains updated links to iMDB for some of the movies in the ml-100k dataset;
        7. links (MovieLens-latest): contains the links to iMDB for the movies in the MovieLens-latest dataset.
        """
        assert os.path.exists(os.path.join(self.data_path, 'mindreader/entities.csv')), "entities.csv for MindReader " \
                                                                                        "is missing"
        assert os.path.exists(os.path.join(self.data_path, 'mindreader/ratings.csv')), "ratings.csv for MindReader " \
                                                                                       "is missing"
        assert os.path.exists(os.path.join(self.data_path, 'mindreader/triples.csv')), "triples.csv for MindReader " \
                                                                                       "is missing"
        assert os.path.exists(os.path.join(self.data_path, 'ml-100k/u.data')), "u.data for ml-100k is missing"
        assert os.path.exists(os.path.join(self.data_path, 'ml-100k/u.item')), "u.item for ml-100k is missing"
        assert os.path.exists(os.path.join(self.data_path, 'ml-latest-small/movies.csv')), "movies.csv for " \
                                                                                           "movielens " \
                                                                                           "latest is missing"
        assert os.path.exists(os.path.join(self.data_path, 'ml-latest-small/links.csv')), "links.csv for " \
                                                                                          "movielens " \
                                                                                          "latest is missing"

    def process_ml_100k(self, threshold=4):
        """
        It removes duplicates in the ml-100k dataset. Some movies appear with different indexes because the same user
        has rated multiple times the same movie (sometimes even with the same rating).
        It also recreates the indexing after these modifications have been done, in such a way the indexes are 
        in [0, n_movies].
        It creates new csv files without the duplicates, both u.data and u.item, with the new indexing.

        :param threshold: a threshold used to create implicit feedbacks from the explicit feedbacks of ml-100k. Ratings
        above the threshold are converted to 1, the others to 0.
        """
        # get MovieLens-100k movies
        ml_100_movies_file = pd.read_csv(os.path.join(self.data_path, "ml-100k/u.item"), sep="|",
                                         encoding="iso-8859-1", header=None)
        ml_100_movies_file_dict = ml_100_movies_file.to_dict("records")

        # for each title, we get the list of corresponding indexes (for some titles we have multiple indexes)
        ml_100_title_to_idx = {}
        for movie in ml_100_movies_file_dict:
            if movie[1] not in ml_100_title_to_idx:
                ml_100_title_to_idx[movie[1]] = [movie[0]]
            else:
                ml_100_title_to_idx[movie[1]].append(movie[0])

        # get movies with multiple indexes
        duplicates = {movie: idx for movie, idx in ml_100_title_to_idx.items() if len(idx) > 1}
        # create new ratings and movie files with new indexes
        ml_ratings = pd.read_csv(os.path.join(self.data_path, "ml-100k/u.data"), sep="\t", header=None)
        new_ratings = []
        for movie, idx in duplicates.items():
            # create one unique index for the same movie with two different indexes (each movie has max two indexes)
            # we keep the first of the two indexes
            ml_ratings[1] = ml_ratings[1].replace(idx[1], idx[0])
            # delete record with second occurrence of the movie in the file containing movie information
            ml_100_movies_file = ml_100_movies_file[ml_100_movies_file[0] != idx[1]]
            # aggregate ratings of users (we need to do that because now we have a user giving multiple
            # ratings to the same index)
            groups = ml_ratings.groupby(by=[0, 1])
            for i, (_, group) in enumerate(groups):
                if len(group) > 1:  # it means we have multiple ratings by the same user to the same item
                    # compute mean of the ratings
                    group[2] = group[2].mean()
                    # append new record to dataset
                    new_ratings.append(group.head(1))
                    # remove old records from dataset
                    ml_ratings = ml_ratings.drop(group.index)
        new_ratings = pd.concat(new_ratings)
        ml_ratings = ml_ratings.append(new_ratings)
        ml_ratings.reset_index()
        ml_100_movies_file.reset_index()
        # create new indexing for users and items of MovieLens-100k
        user_mapping = {}
        item_mapping = {}
        u = i = 0
        ml_ratings_dict = ml_ratings.to_dict("records")
        for rating in ml_ratings_dict:
            if rating[0] not in user_mapping:
                user_mapping[rating[0]] = u
                u += 1
            if rating[1] not in item_mapping:
                item_mapping[rating[1]] = i
                i += 1
            rating[0] = user_mapping[rating[0]]
            rating[1] = item_mapping[rating[1]]
            rating[2] = 1 if rating[2] >= threshold else 0

        ml_ratings = pd.DataFrame.from_records(ml_ratings_dict)
        # create correct indexes also for the file containing the movie information
        new_indexes = []
        for _, movie_idx in ml_100_movies_file[0].items():
            new_indexes.append(item_mapping[movie_idx])
        ml_100_movies_file[0] = new_indexes
        # sort movies by index for a better reading of file
        ml_100_movies_file = ml_100_movies_file.sort_values(by=[0])
        ml_100_movies_file.reset_index()

        # get genres of the movies - they are one-hot encoded starting from column at index 6 of every record
        ml_100_movies_records = []
        for row_idx, row in ml_100_movies_file.iterrows():
            # 0 means the item has not the genre, 1 means it has the genre
            movie_genres = [genre for _, genre in row[6:].items()]
            # get indexes of the genres that the item belongs to - these indexes follow the order in self.genres
            # which is also the order specified in the readME of the ml-100k dataset
            movie_genres_idx = [str(idx) for idx, genre in enumerate(movie_genres) if genre == 1]
            # we specify None when an item has no genres - the genres are specified with their indexes according to the
            # order in self.genres
            ml_100_movies_records.append({"idx": row[0], "title": row[1], "genres": "|".join(movie_genres_idx)
            if movie_genres_idx else "None"})

        # create new files
        ml_ratings.to_csv(os.path.join(self.data_path, "ml-100k/processed/u.data"), sep="\t", index=False, header=None)
        ml_100_movies_file = pd.DataFrame.from_records(ml_100_movies_records)
        ml_100_movies_file.to_csv(os.path.join(self.data_path, "ml-100k/processed/u.item"),
                                  encoding="iso-8859-1", index=False, header=None)

        # save also the files containing the mappings
        user_mapping_records = [{"old_idx": old_idx, "new_idx": new_idx} for old_idx, new_idx in user_mapping.items()]
        item_mapping_records = [{"old_idx": old_idx, "new_idx": new_idx} for old_idx, new_idx in item_mapping.items()]
        user_mapping_file = pd.DataFrame.from_records(user_mapping_records)
        user_mapping_file.to_csv(os.path.join(self.data_path, "ml-100k/processed/user_mapping.csv"), index=False)
        item_mapping_file = pd.DataFrame.from_records(item_mapping_records)
        item_mapping_file.to_csv(os.path.join(self.data_path, "ml-100k/processed/item_mapping.csv"), index=False)

    def process_mr(self):
        """
        It processes the MindReader dataset by removing duplicated titles. There could be cases in which the title is
        the same even if the movies are slightly different, for example they have a different publication year.
        To remove duplicates, we add the publication date to the title of the movies (in brackets), as for the MovieLens
        datasets.
        After these modifications have been done, it recreates the indexing of MindReader by substituting entity URIs
        with proper indexes in [0, n_movies]. The same is done for the user identifiers, which are also represented
        using strings in MindReader.
        This procedure also filters the dataset by removing the unknown ratings (ratings at 0) and changing negative
        ratings from -1 to 0.
        At the end of the procedure, new files with proper modifications and indexing are created.
        """
        mr_entities = pd.read_csv(os.path.join(self.data_path, "mindreader/entities.csv"))
        mr_entities_dict = mr_entities.to_dict("records")
        # get mapping from URI to title for the movies of MindReader
        mr_uri_to_title = {entity["uri"]: entity["name"] for entity in mr_entities_dict
                           if "Movie" in entity["labels"]}
        # get publication dates of movies of MindReader by querying Wikidata
        query = "SELECT ?film ?publicationDate WHERE { ?film wdt:P577 ?publicationDate. VALUES ?film {" + \
                ' '.join(["wd:" + uri.split("/")[-1] for uri in mr_uri_to_title]) + \
                "} SERVICE wikibase:label { bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en'. } }"

        result = send_query(query)

        # create a mapping from URI to publication date for the movies of MindReader
        mr_uri_to_date = {}
        for match in result:
            uri = match['film']['value']
            date = match['publicationDate']['value'].split("-")[0]  # we need just the publication YEAR
            if uri not in mr_uri_to_date:
                # we take only the first date for each movie, some movies have multiple publication dates
                mr_uri_to_date[uri] = date

        # put date on the title (in brackets) when the date is available
        # in the case the date is not available, we keep just the title of the movie
        mr_uri_to_title = {uri: "%s (%d)" % (title, int(mr_uri_to_date[uri]))
        if uri in mr_uri_to_date else title for uri, title in mr_uri_to_title.items()}
        # recreate "name" column of original dataframe - the new column has now dates on the titles when they are
        # available
        # Attention: the entities file does not contain just movies, but all the entities of MindReader
        # for this reason we also check if the entity is a movie or not
        mr_entities["name"] = [mr_uri_to_title[entity["uri"]]
                               if "Movie" in entity["labels"] else entity["name"] for entity in mr_entities_dict]
        # create new indexing for the dataset (users, movies)
        mr_ratings = pd.read_csv(os.path.join(self.data_path, "mindreader/ratings.csv"))
        mr_entities_dict = mr_entities.to_dict("records")
        mr_movies = [entity["uri"] for entity in mr_entities_dict if "Movie" in entity["labels"]]
        # we want ratings only for the allowed genres - we create a mapping between URIs and genre names
        # for allowed genres
        mr_genres = {entity["uri"]: entity["name"] for entity in mr_entities_dict if "Genre" in entity["labels"]
                     if entity["name"] in self.genres}
        # remove unknown ratings
        mr_movie_ratings = mr_ratings[
            mr_ratings["uri"].isin(mr_movies) & mr_ratings["sentiment"] != 0].reset_index().to_dict("records")
        mr_genre_ratings = mr_ratings[
            mr_ratings["uri"].isin(mr_genres) & mr_ratings["sentiment"] != 0].reset_index().to_dict("records")
        u = i = 0
        user_mapping = {}
        item_mapping = {}
        mr_movie_ratings_new = []
        for rating in mr_movie_ratings:
            if rating["userId"] not in user_mapping:
                user_mapping[rating["userId"]] = u
                u += 1
            if rating["uri"] not in item_mapping:
                item_mapping[rating["uri"]] = i
                i += 1
            rating["uri"] = item_mapping[rating["uri"]]
            rating["userId"] = user_mapping[rating["userId"]]
            # set negative ratings to 0 and positive ratings to 1
            rating["sentiment"] = int(rating["sentiment"] > 0)
            mr_movie_ratings_new.append(
                {"u_idx": rating["userId"], "i_idx": rating["uri"], "rate": rating["sentiment"]})
        mr_movie_ratings_new = pd.DataFrame.from_records(mr_movie_ratings_new)
        mr_movie_ratings_new.to_csv(os.path.join(self.data_path, "mindreader/processed/movie_ratings.csv"), index=False)

        mr_genre_ratings_new = []
        for rating in mr_genre_ratings:
            # some users could have rated only genres of movies - for this reason they could not be in the
            # user mapping yet
            if rating["userId"] not in user_mapping:
                user_mapping[rating["userId"]] = u
                u += 1
            rating["userId"] = user_mapping[rating["userId"]]
            # get index of the genre in the self.genres list - this list is used both for ml-100k and MindReader
            # to unify the indexing of the genres
            rating["uri"] = self.genres.index(mr_genres[rating["uri"]])
            rating["sentiment"] = int(rating["sentiment"] > 0)
            mr_genre_ratings_new.append(
                {"u_idx": rating["userId"], "g_idx": rating["uri"], "rate": rating["sentiment"]})
        mr_genre_ratings_new = pd.DataFrame.from_records(mr_genre_ratings_new)
        mr_genre_ratings_new.to_csv(os.path.join(self.data_path, "mindreader/processed/genre_ratings.csv"), index=False)

        # now, we need to associate each movie to its genres - we use a dict for doing that
        mr_triples_dict = pd.read_csv(os.path.join(self.data_path, "mindreader/triples.csv")).to_dict("records")
        mr_movie_genre_dict = {}
        for triple in mr_triples_dict:
            # consider only mapped movies and allowed genres
            if triple["head_uri"] in item_mapping and triple["tail_uri"] in mr_genres \
                    and triple["relation"] == "HAS_GENRE":
                if item_mapping[triple["head_uri"]] not in mr_movie_genre_dict:
                    mr_movie_genre_dict[item_mapping[
                        triple["head_uri"]]] = [self.genres.index(mr_genres[triple["tail_uri"]])]
                else:
                    mr_movie_genre_dict[item_mapping[
                        triple["head_uri"]]].append(self.genres.index(mr_genres[triple["tail_uri"]]))

        # now, we need to create a file containing movie information, as for MovieLens
        # the file will contain also the genres of the movies, as in MovieLens
        # if a movie is not in mr_movie_genre_dict, it means that the movie has not ratings, so we do not include it
        # in the entities since it is not necessary for the purpose of our experiments
        # there could be a movie that is rated but does not belong to any of the selected genres. In such a case
        # we set the genres of the movie to None
        mr_entities_new = []
        for entity in mr_entities_dict:
            # if the uri is in item_mapping, it means that it corresponds to a movie and it has been rated
            if entity["uri"] in item_mapping:
                mr_entities_new.append({"idx": item_mapping[entity["uri"]],
                                        "title": entity["name"],
                                        "genres": "|".join([str(g)
                                                            for g in mr_movie_genre_dict[item_mapping[entity["uri"]]]
                                                            ])
                                        if item_mapping[entity["uri"]] in mr_movie_genre_dict else "None"})
        mr_entities_new = pd.DataFrame.from_records(mr_entities_new)
        # sort file by index for better readability
        mr_entities_new = mr_entities_new.sort_values(by=["idx"])
        mr_entities_new.reset_index()
        mr_entities_new.to_csv(os.path.join(self.data_path, "mindreader/processed/movies.csv"), index=False)

        # save also the files containing the mappings
        user_mapping_records = [{"old_idx": old_idx, "new_idx": new_idx} for old_idx, new_idx in user_mapping.items()]
        item_mapping_records = [{"old_idx": old_idx, "new_idx": new_idx} for old_idx, new_idx in item_mapping.items()]
        user_mapping_file = pd.DataFrame.from_records(user_mapping_records)
        user_mapping_file.to_csv(os.path.join(self.data_path, "mindreader/processed/user_mapping.csv"), index=False)
        item_mapping_file = pd.DataFrame.from_records(item_mapping_records)
        item_mapping_file.to_csv(os.path.join(self.data_path, "mindreader/processed/item_mapping.csv"), index=False)

    def get_mr_200k_dataset(self, seed, binary_ratings=True, genre_threshold=None, k_filter_genre=None,
                            k_filter_movie=None,
                            genre_val_size=0.2, val_mode="auc", genre_user_level_split=False, movie_val_size=None,
                            movie_test_size=None, movie_user_level_split=False, n_neg=None, true_neg_rank=False,
                            implicit_feedback=False):
        """
        It returns the MindReader-200k dataset ready for learning. The dataset is subdivided as follows:
        1. genre folds: a training set and validation set for learning the genre preferences of the users;
        2. item folds: a training set of user-item pairs and a validation and test sets for computing AUC. In particular,
        for each user, one random positive and negative interactions are randomly sampled for constructing validation
        and test sets.

        :param seed: seed for random reproducibility
        :param binary_ratings: whether the ratings has to be binarized in [0,1]
        :param genre_threshold: number of genres to be kept in the dataset. If None, all the genres are kept. If an
        integer is given, only the `genre_threshold` most popular genres are kept.
        :param k_filter_movie: threshold to filter underpopulated users and genres
        :param genre_val_size: proportion of validation interactions for constructing the validation set of the genre
        dataset
        :param val_mode: str containing the validation mode. The validation mode could be 1 plus random, AUC, or
        rating prediciton.
        :param movie_val_size: proportion of validation interactions for constructing the validation set of the movie
        dataset. Defaults to None. If None, it means the validation data has to be constructed for ranking prediction.
        If it is different from None, it means the validation data has to be constructed for rating prediction.
        :param genre_user_level_split: whether the split for user-genre ratings has to be performed at the user level
        or not
        :param movie_user_level_split: whether the split for user-item ratings has to be performed at the user level
        or not
        :param n_neg: number of negative items that have to be randomly sampled for each user to create the fold for
        ranking computation. Defaults to None, meaning that the fold created is for AUC computation.
        :param true_neg_rank: whether the negative items randomly sampled for each validation example for the ranking
        task have to be sampled from the negative items of the user or from the non-relevant items. Defaults to False,
        meaning that they are sampled from the non-relevant items (non interacted items).
        :param implicit_feedback: whether the dataset has to be converted to implicit feedback or not. In the case it
        is converted, only the positive ratings are kept. Defaults to False, meaning the dataset is left explicit. This
        conversion is applied only to movie ratings.
        :return: two datasets, one for learning genre preferences and one for learning movie preferences
        """
        assert val_mode in ["1+random", "auc", "rating-prediction"], "The selected validation mode is not available"
        if implicit_feedback and val_mode == "rating-prediction":
            raise ValueError("You cannot select implicit feedback and rating prediction as validation mode.")
        # get MindReader entities
        mr_entities = pd.read_csv(os.path.join(self.data_path, "mindreader-200k/entities.csv"))
        mr_entities_dict = mr_entities.to_dict("records")
        # get mr_genres
        mr_genres = [entity["uri"] for entity in mr_entities_dict if "Genre" in entity["labels"]]
        # get MindReader ratings
        mr_ratings = pd.read_csv(os.path.join(self.data_path, "mindreader-200k/ratings.csv"))
        # remove useless columns from dataframe
        mr_ratings = mr_ratings.drop(mr_ratings.columns[[0, 3]], axis=1).reset_index(drop=True)
        # get ratings on movie genres and remove unknown ratings
        genre_ratings = mr_ratings[mr_ratings["uri"].isin(mr_genres) & mr_ratings["sentiment"] != 0]
        # get ratings only for the most rated genres, if requested
        if genre_threshold is not None:
            # get the top `genre_threshold` most rated genres
            most_rated_genres = list(genre_ratings.groupby(["uri"]).size().reset_index(
                name='counts').sort_values(by=["counts"], ascending=False).head(genre_threshold)["uri"])
            # get ratings only for the most rated genres
            genre_ratings = genre_ratings[genre_ratings["uri"].isin(most_rated_genres)]

        def filter_ratings(ratings, k_filter):
            if k_filter is not None:
                tmp1 = ratings.groupby(['userId'], as_index=False)['uri'].count()
                tmp1.rename(columns={'uri': 'cnt_item'}, inplace=True)
                tmp2 = ratings.groupby(['uri'], as_index=False)['userId'].count()
                tmp2.rename(columns={'userId': 'cnt_user'}, inplace=True)
                ratings = ratings.merge(tmp1, on=['userId']).merge(tmp2, on=['uri'])
                ratings = ratings.query(f'cnt_item >= {k_filter} and cnt_user >= {k_filter}')
                ratings = ratings.drop(['cnt_item', 'cnt_user'], axis=1)
                del tmp1, tmp2
            return ratings

        # filter genres rated by less than `k_filter_genre` users and users who rated less than 5 genres to make
        # movie genre ratings less sparse
        genre_ratings = filter_ratings(genre_ratings, k_filter_genre)

        # get movie ratings and remove unknown ratings
        mr_movies = [entity["uri"] for entity in mr_entities_dict if "Movie" in entity["labels"]]
        movie_ratings = mr_ratings[mr_ratings["uri"].isin(mr_movies) & mr_ratings["sentiment"] != 0]
        # filter users who rated less than `k_filter_movie` movies and movies with less than `k_filter_movie` ratings
        movie_ratings = filter_ratings(movie_ratings, k_filter_movie)
        if implicit_feedback:
            # remove negative ratings in such a way they are treated as non-relevant items together
            # with the non-interacted items
            movie_ratings = movie_ratings[movie_ratings["sentiment"] == 1]
        # get ids of users who rated at least 'k_filter' genre
        u_who_rated_genres = genre_ratings["userId"].unique()
        # get ids of users who rated at least 'k_filter' movies
        u_who_rated_movies = movie_ratings["userId"].unique()
        # get intersection of previous users to get the set of users of the dataset - we want the same exact set of
        # users on both datasets (the genre one and the movie one) since we need to perform transfer learning and
        # knowledge transfer
        u_who_rated = set(u_who_rated_movies) & set(u_who_rated_genres)
        # remove all movie ratings of users who did not rate at least 'k_filter' genres
        movie_ratings = movie_ratings[movie_ratings["userId"].isin(u_who_rated)]
        # remove all genre ratings of user who did not rate at least 'k_filter' movies
        genre_ratings = genre_ratings[genre_ratings["userId"].isin(u_who_rated)]
        # get integer user, genre, and item indexes
        u_ids = genre_ratings["userId"].unique()
        int_u_ids = list(range(len(u_ids)))
        user_string_to_id = dict(zip(u_ids, int_u_ids))
        g_ids = genre_ratings["uri"].unique()
        int_g_ids = list(range(len(g_ids)))
        g_string_to_id = dict(zip(g_ids, int_g_ids))
        i_ids = movie_ratings["uri"].unique()
        int_i_ids = list(range(len(i_ids)))
        i_string_to_id = dict(zip(i_ids, int_i_ids))
        genre_ratings = genre_ratings.replace(user_string_to_id).replace(g_string_to_id).reset_index(drop=True)
        movie_ratings = movie_ratings.replace(user_string_to_id).replace(i_string_to_id).reset_index(drop=True)

        # convert negative ratings from -1 to 0
        if binary_ratings and not implicit_feedback:
            genre_ratings = genre_ratings.replace(-1, 0)
            movie_ratings = movie_ratings.replace(-1, 0)

        # get number of users and genres and movies
        n_users = genre_ratings["userId"].nunique()
        n_genres = genre_ratings["uri"].nunique()
        n_items = movie_ratings["uri"].nunique()

        # compute user-item sparse matrix
        u_i_matrix = csr_matrix((np.ones(len(movie_ratings)), (movie_ratings["userId"], movie_ratings["uri"])),
                                shape=(n_users, n_items))

        def train_test_split(ratings, frac=None, user_level=False):
            """
            It splits the dataset into training and test sets.

            :param ratings: dataframe containing the dataset ratings that have to be split
            :param frac: proportion of positive ratings to be sampled. If None, it randomly sample one positive
            rating (LOO)
            :param user_level: whether the train test split has to be performed at the user level or ratings can
            be sampled randomly independently from the user. Defaults to True, meaning that the split is done at the
            user level. In case the split is not at the user level, a stratified split is performed.
            :return: train and test set dataframes
            """
            if user_level:
                test_ids = ratings.groupby(by=["userId"]).apply(
                    lambda x: x.sample(frac=frac, random_state=seed).index
                ).explode().values
                # remove NaN indexes - when pandas is not able to sample due to small group size and high frac,
                # it returns an empty index which then becomes NaN when converted in numpy
                test_ids = test_ids[~np.isnan(test_ids.astype("float64"))]
                train_ids = np.setdiff1d(ratings.index.values, test_ids)
                train_set = ratings.iloc[train_ids].reset_index(drop=True)
                test_set = ratings.iloc[test_ids].reset_index(drop=True)
                return train_set, test_set
            else:
                # we use scikit-learn train-test split
                return train_test_split_sklearn(ratings, random_state=seed, stratify=ratings["sentiment"],
                                                test_size=frac)

        def create_folds_for_auc_computation(item_ratings):
            """
            It randomly samples one positive and one negative rating for each user. These are used as validation
            ratings to compute AUC metric.

            :param item_ratings: dataset of ratings from which the interactions have to be sampled for creating the
            validation or test set
            :return: a dataframe containing the remaining ratings (training set) and a np array of
            (user, pos_item, neg_item) triples for computing AUC
            """
            # todo questa funzione e' un piccolo collo di bottiglia
            # group by userId
            users = item_ratings.groupby(by=["userId"])
            auc_triples = []
            for user, user_ratings in users:
                # check if the user has at least one positive rating and one negative rating
                pos_ratings = user_ratings[user_ratings["sentiment"] == 1]
                if not implicit_feedback:
                    neg_ratings = user_ratings[user_ratings["sentiment"] != 1]
                    if len(pos_ratings) > 1 and len(neg_ratings) > 1:
                        # sample one positive and one negative rating for this user
                        pos_int = pos_ratings.sample(random_state=seed)
                        neg_int = neg_ratings.sample(random_state=seed)
                        auc_triples.append([[user, int(neg_int["uri"])], [user, int(pos_int["uri"])]])
                        # drop positive and negative interactions from original dataframe
                        item_ratings.drop(pos_int.index, inplace=True)
                        item_ratings.drop(neg_int.index, inplace=True)
                else:
                    if len(pos_ratings) > 1:
                        # sample one positive rating for this user
                        pos_int = pos_ratings.sample(random_state=seed)
                        # sample one non-relevant item for this user
                        neg_int = random.choice(list(set(range(n_items)) - set(u_i_matrix[user].nonzero()[1])))
                        # create AUC validation example for the current user
                        auc_triples.append([[user, neg_int], [user, int(pos_int["uri"])]])
                        # drop positive interaction from original dataframe
                        item_ratings.drop(pos_int.index, inplace=True)
            return item_ratings, np.array(auc_triples)

        def create_fold_for_ranking_computation(item_ratings):
            """
            It takes the given dataframe of ratings as input and creates the validation/test fold for ranking
            computation.

            For each user, one random positive item is held-out for validation/test set, then `n_neg` random
            non-relevant items are sampled from the set of items that the user did not see.

            :param item_ratings: dataframe containing the ratings from which the validation fold has to be created
            :return: dataframe containing the training ratings and a np.array containing the validation/test fold.
            For each user, the positive item is put at the end of the item list.
            """
            # group by userId
            users = item_ratings.groupby(by=["userId"])
            val_fold = []
            for user, user_ratings in users:
                # check if the user has at least one positive rating
                pos_ratings = user_ratings[user_ratings["sentiment"] == 1]
                if len(pos_ratings) > 1:
                    # sample one positive rating
                    pos_int = pos_ratings.sample(random_state=seed)
                    if true_neg_rank:
                        # take the negative ratings of the user
                        neg_ratings = user_ratings[user_ratings["sentiment"] != 1]["uri"]
                        # check if there are enough negative ratings and that the number of sampled ratings is less than
                        # 40% of the total number of negative ratings of the user
                        if len(neg_ratings) > n_neg and (n_neg / len(neg_ratings)) <= 0.4:
                            random_negs = neg_ratings.sample(n=n_neg, random_state=seed)
                            # remove the sampled ratings from the original dataframe
                            item_ratings.drop(random_negs.index, inplace=True)
                            random_negs = list(random_negs)
                        else:
                            # we pass to the next user if we do not have enough neg ratings to sample for this user
                            continue
                    else:
                        # sample n_neg non-relevant items for this user
                        neg_ints = list(set(range(n_items)) - set(u_i_matrix[user].nonzero()[1]))
                        random_negs = random.sample(neg_ints, n_neg)
                    # create validation example for the current user
                    val_fold.append([[user, neg] for neg in random_negs] + [[user, int(pos_int["uri"])]])
                    # drop positive interactions from original dataframe
                    item_ratings.drop(pos_int.index, inplace=True)
            return item_ratings, np.array(val_fold)

        # create train and validation set for genre preferences
        g_train_set, g_val_set = train_test_split(genre_ratings, frac=genre_val_size, user_level=genre_user_level_split)
        # create train, validation and test set for movie preferences according to the selected task
        val_examples, test_examples = None, None
        if val_mode == "rating-prediction":
            assert movie_val_size is not None and movie_test_size is not None, "You selected rating " \
                                                                               "prediction as validation mode, so you" \
                                                                               "must specify the proportion of ratings" \
                                                                               "to be put in validation and test sets."
            train_set, test_set = train_test_split(movie_ratings, frac=movie_test_size,
                                                   user_level=movie_user_level_split)
            train_set_small, val_set = train_test_split(train_set, frac=movie_test_size,
                                                        user_level=movie_user_level_split)
        elif val_mode == "1+random":
            assert n_neg is not None, "You selected one plus random as validation mode, so you must specify parameter" \
                                      "n_neg, which indicates the number of negative items for each validation example."
            train_set, test_examples = create_fold_for_ranking_computation(movie_ratings)
            train_set_small, val_examples = create_fold_for_ranking_computation(train_set)
        elif val_mode == "auc":
            train_set, test_examples = create_folds_for_auc_computation(movie_ratings)
            train_set_small, val_examples = create_folds_for_auc_computation(train_set)

        # create numpy arrays of user-item ratings with interaction matrices
        def create_fold(fold, n_items):
            """
            It creates a dataset fold ready for the training or testing of the model.

            :param fold: dataframe containing user-item interactions
            :param n_items: number of items in the dataset, used to create the user-item sparse matrix. It has to be
            passed because depending on the fold, n_items can be the number of movies or the number of genres.
            :return: a dictionary. The first key (ratings) is a numpy array containing user-item-rating triples, the
            second one (interaction_matrix) is a
            csr sparse matrix containing the user-item interactions. A 1 in the matrix means that the user
            interacted with the item (it gave a positive or negative rating). A 0 means that is is an unobserved item
            for the user. The third key is the rating matrix. Same shape of the interaction matrix but filled with
            ground truth.
            """
            return {"ratings": np.array([tuple(rating.values()) for rating in fold.to_dict("records")]),
                    "interaction_matrix": csr_matrix((np.ones(len(fold)), (list(fold["userId"]), list(fold["uri"]))),
                                                     shape=(n_users, n_items)),
                    "rating_matrix": csr_matrix((np.array([rating["sentiment"] for rating in fold.to_dict("records")]),
                                                 (list(fold["userId"]), list(fold["uri"]))), shape=(n_users, n_items))}

        genre_folds = {"train_set": create_fold(g_train_set, n_genres),
                       "val_set": create_fold(g_val_set, n_genres)}

        movie_folds = {"entire_dataset": u_i_matrix,
                       "train_set": create_fold(train_set, n_items),
                       "train_set_small": create_fold(train_set_small, n_items),
                       "val_set": val_examples if val_examples is not None else create_fold(val_set, n_items),
                       "test_set": test_examples if test_examples is not None else create_fold(test_set, n_items)}

        # create itemXgenres matrix
        # get MindReader triples file
        mr_triples = pd.read_csv("./datasets/mindreader-200k/triples.csv")
        # get only triples with HAS_GENRE relationship, where the head is a movie and the tail a movie genre
        mr_genre_triples = mr_triples[mr_triples["head_uri"].isin(i_string_to_id.keys()) &
                                      mr_triples["tail_uri"].isin(g_string_to_id.keys()) &
                                      (mr_triples["relation"] == "HAS_GENRE")]
        # replace URIs with integer indexes
        mr_genre_triples = mr_genre_triples.replace(i_string_to_id).replace(g_string_to_id)
        # construct the itemXgenres matrix
        # todo alcuni film non hanno generi associati e c'e' un genere che non e' associato ad alcun film
        item_genres_matrix = csr_matrix((np.ones(len(mr_genre_triples)),
                                         (list(mr_genre_triples["head_uri"]),
                                          list(mr_genre_triples["tail_uri"]))), shape=(n_items, n_genres))

        return n_users, n_genres, n_items, genre_folds, movie_folds, torch.tensor(item_genres_matrix.toarray())

    def get_mr_100k_genre_ratings(self):
        # get MindReader-100k entities
        mr_entities = pd.read_csv(os.path.join(self.data_path, "mindreader/entities.csv"))
        mr_entities_dict = mr_entities.to_dict("records")
        # get mr_genres
        mr_genres = [entity["uri"] for entity in mr_entities_dict if "Genre" in entity["labels"]]
        # get MindReader ratings
        mr_ratings = pd.read_csv(os.path.join(self.data_path, "mindreader/ratings.csv"))
        # get ratings on movie genres and remove unknown ratings
        return mr_ratings[mr_ratings["uri"].isin(mr_genres) & mr_ratings["sentiment"] != 0]

    def get_mr_genre_ratings(self, seed, genre_threshold=None, binary_ratings=True, k_filter=None, test_size=0.2, val_size=0.1, genre_val_size=0.2):
        """
        It creates the folds for training, validating, and testing a recommendation model on the MindReader dataset.
        Specifically, it follows the following procedure:
        1. take the movie genre ratings from MindReader, different from unknown
        2. filter user who rated less than k genres and genres that have been rated by less than 5 users
        3. change indexing of users and genres in such a way they are progressive integers
        4. take the user-movie ratings from MindReader (different from unknown) by filtering out all the users that are
        not present in the user-genre fold. We want only the users who rated at least one genre
        5. change indexing of users and movies in such a way they are progressive integers
        6. perform leave-one-out on the user-genre fold by taking one positive genre rating for each user to create
        the test set. The same is done for creating the validation set from the remaining training ratings
        7. perform a 80% train 20% test split on the user-movie fold. Then 90% train and 10% validation are used as
        proportions to create the validation set from the training set
        8. for each fold, the user-item interaction matrix is created, with users on the rows and items on the columns.
        This matrix has a 1 on all the positions in which the user interacted with the item and a 0 otherwise. It is
        a csr sparse matrix.

        :param seed: seed for random reproducibility
        :param genre_threshold: threshold to filter the genre ratings in such a way that only the most rated genres are
        kept. This is done to reduce sparsity. The genres are sorted according to the number of ratings and then only
        the top `genre_threshold` genres are kept
        :param binary_ratings: whether the ratings have to be converted from [-1, 1] to [0, 1], or not
        :param k_filter: threshold used to filter the users and genres with less than 5 interactions
        :param test_size: proportion of user-movie positive ratings to be sampled from the dataset to construct
        the test set
        :param val_size: proportion of user-movie positive ratings to be sampled from the training set to construct the
        validation set
        :return a tuple of 5 elements. Number of users, number of movies, number of genres, a tuple with the folds for
        movie ratings (4 folds, train set complete, train set small, val set, test set), a tuple with the folds for
        genre ratings (same as movie ratings)
        """
        # get MindReader entities
        mr_entities = pd.read_csv(os.path.join(self.data_path, "mindreader/entities.csv"))
        mr_entities_dict = mr_entities.to_dict("records")
        # get mr_genres
        mr_genres = [entity["uri"] for entity in mr_entities_dict if "Genre" in entity["labels"]]
        # get MindReader ratings
        mr_ratings = pd.read_csv(os.path.join(self.data_path, "mindreader/ratings.csv"))
        # get ratings on movie genres and remove unknown ratings
        genre_ratings = mr_ratings[mr_ratings["uri"].isin(mr_genres) & mr_ratings["sentiment"] != 0]
        # convert negative ratings from -1 to 0
        if binary_ratings:
            genre_ratings = genre_ratings.replace(-1, 0)
        if genre_threshold is not None:
            # get the top `genre_threshold` most rated genres
            most_rated_genres = list(genre_ratings.groupby(["uri"]).size().reset_index(
                name='counts').sort_values(by=["counts"], ascending=False).head(genre_threshold)["uri"])
            # the ratings only for the most rated genres
            genre_ratings = genre_ratings[genre_ratings["uri"].isin(most_rated_genres)]
        # filter genres rated by less than 5 users and users who rated less than 5 genres to make movie genre
        # ratings less sparse
        if k_filter is not None:
            tmp1 = genre_ratings.groupby(['userId'], as_index=False)['uri'].count()
            tmp1.rename(columns={'uri': 'cnt_item'}, inplace=True)
            tmp2 = genre_ratings.groupby(['uri'], as_index=False)['userId'].count()
            tmp2.rename(columns={'userId': 'cnt_user'}, inplace=True)
            genre_ratings = genre_ratings.merge(tmp1, on=['userId']).merge(tmp2, on=['uri'])
            genre_ratings = genre_ratings.query(f'cnt_item >= {k_filter} and cnt_user >= {k_filter}')
            genre_ratings = genre_ratings.drop(['cnt_item', 'cnt_user'], axis=1)
            del tmp1, tmp2
        # get ratings on movies - remove unknown ratings and convert negatives from -1 to 0
        mr_movies = [entity["uri"] for entity in mr_entities_dict if "Movie" in entity["labels"]]
        movie_ratings = mr_ratings[mr_ratings["uri"].isin(mr_movies) & mr_ratings["sentiment"] != 0]
        if binary_ratings:
            movie_ratings = movie_ratings.replace(-1, 0)
        # get users who rated at least 5 genres
        users_rated_k_genres = genre_ratings["userId"].unique()
        # remove all the other users from the movie ratings
        movie_ratings = movie_ratings[movie_ratings["userId"].isin(users_rated_k_genres)]
        # get integer user, genre and item indexes
        u_ids = genre_ratings["userId"].unique()
        int_u_ids = list(range(len(u_ids)))
        user_string_to_id = dict(zip(u_ids, int_u_ids))
        g_ids = genre_ratings["uri"].unique()
        int_g_ids = list(range(len(g_ids)))
        g_string_to_id = dict(zip(g_ids, int_g_ids))
        i_ids = movie_ratings["uri"].unique()
        int_i_ids = list(range(len(i_ids)))
        item_string_to_id = dict(zip(i_ids, int_i_ids))
        genre_ratings = genre_ratings.replace(user_string_to_id).replace(g_string_to_id).reset_index(drop=True)
        movie_ratings = movie_ratings.replace(user_string_to_id).replace(item_string_to_id).reset_index(drop=True)
        # remove useless columns
        genre_ratings = genre_ratings.drop(genre_ratings.columns[[0, 1, 4]], axis=1).reset_index(drop=True)
        movie_ratings = movie_ratings.drop(movie_ratings.columns[[0, 1, 4]], axis=1).reset_index(drop=True)
        # get number of users, genres, and items
        n_users = genre_ratings["userId"].nunique()
        n_genres = genre_ratings["uri"].nunique()
        n_items = movie_ratings["uri"].nunique()

        # random sample one positive item for each user to create test set
        def train_test_split(ratings, frac=None, genres=False):
            """
            It splits the dataset into training and test sets.

            :param ratings: dataframe containing the dataset ratings that have to be split
            :param frac: proportion of positive ratings to be sampled. If None, it randomly sample one positive
            rating (LOO)
            :return: train and test set dataframes containing the positive user-item interactions randomly sampled
            by the procedure
            """
            # filter the ratings in such a way that only the positives remain - we are interested in having positive
            # ratings in validation. We want to test how the recommendation places the target positive items
            # todo rimosso perche' non ci interessa fare ranking a noi sui generi [ratings["sentiment"] == 1]
            test_ids = (ratings[ratings["sentiment"] == 1] if not genres else ratings).groupby(by=["userId"]).apply(
                lambda x: x.sample(frac=frac, random_state=seed).index
            ).explode().values
            # remove NaN indexes - when pandas is not able to sample due to small group size and high frac, it returns
            # an empty index which then becomes NaN when converted in numpy
            test_ids = test_ids[~np.isnan(test_ids.astype("float64"))]
            train_ids = np.setdiff1d(ratings.index.values, test_ids)
            train_set = ratings.iloc[train_ids].reset_index(drop=True)
            test_set = ratings.iloc[test_ids].reset_index(drop=True)
            return train_set, test_set

        train_set, val_set = train_test_split(genre_ratings, frac=genre_val_size, genres=True)

        # create numpy arrays of user-item ratings with interaction matrices
        def create_fold(fold, n_items):
            """
            It creates a dataset fold ready for the training or testing of the model.

            :param fold: dataframe containing user-item interactions
            :param n_items: number of items in the dataset, used to create the user-item sparse matrix. It has to be
            passed because depending on the fold, n_items can be the number of movies or the number of genres.
            :return: a dictionary. The first key (ratings) is a numpy array containing user-item-rating triples, the
            second one (matrix) is a
            csr sparse matrix containing the same user-item interactions. A 1 in the matrix means that the user
            interacted with the item (it gave a positive or negative rating). A 0 means that is is an unobserved item
            for the user.
            """
            return {"ratings": np.array([tuple(rating.values()) for rating in fold.to_dict("records")]),
                    "matrix": csr_matrix((np.ones(len(fold)), (list(fold["userId"]), list(fold["uri"]))),
                                         shape=(n_users, n_items))}

        genre_folds = {"train_set": create_fold(train_set, n_genres),
                       "val_set": create_fold(val_set, n_genres)}

        # todo abbiamo un problema di sbilanciamento, abbiamo molti piu' rating negativi rispetto ai positivi,
        #  quindi il modello e' incentivato a concentrarsi sui positivi

        train_set_complete, test_set = train_test_split(movie_ratings, frac=test_size)
        train_set_small, val_set = train_test_split(train_set_complete, frac=val_size)

        item_folds = {"train_set_complete": create_fold(train_set_complete, n_items),
                      "train_set_small": create_fold(train_set_small, n_items),
                      "val_set": create_fold(val_set, n_items),
                      "test_set": create_fold(test_set, n_items)}

        # create itemXgenres matrix
        # get MindReader triples file
        mr_triples = pd.read_csv("./datasets/mindreader/triples.csv")
        # get only triples with HAS_GENRE relationship, where the head is a movie and the tail a movie genre
        mr_genre_triples = mr_triples[mr_triples["head_uri"].isin(item_string_to_id.keys()) &
                                      mr_triples["tail_uri"].isin(g_string_to_id.keys()) &
                                      (mr_triples["relation"] == "HAS_GENRE")]
        # replace URIs with integer indexes
        mr_genre_triples = mr_genre_triples.replace(item_string_to_id).replace(g_string_to_id)
        # construct the itemXgenres matrix
        # todo alcuni film non hanno generi associati e c'e' un genere che non e' associato ad alcun film
        item_genres_matrix = csr_matrix((np.ones(len(mr_genre_triples)),
                                                (list(mr_genre_triples["head_uri"]),
                                                 list(mr_genre_triples["tail_uri"]))), shape=(n_items, n_genres))

        return n_users, n_items, n_genres, item_folds, genre_folds, item_genres_matrix, \
               {"user": {v: k for k, v in user_string_to_id.items()}, "genre": {v: k for k, v in g_string_to_id.items()}}

    def create_mapping(self):
        """
        It matches the movies of ml-100k with the movies of MindReader to find the joint set of the final dataset.

        For doing that, it matches titles and years of the movies (the publication date is assumed to be after the
        title and between brackets (the pre-processing prepare the dataset in such a manner)).
        In the cases in which there is not a perfect match between the strings (==), it tries to use fuzzy methods
        to check the similarity (~=) between the titles.

        The process is summarized as follows:
        1. the titles and years of MovieLens-100k are matched with titles and years of MovieLens-latest-small. This
        is done because the latter dataset has updated links to iMDB entities for some of the movies in ml-100k;
        2. the iMDB ids that have been found at the previous step are given as input to a sparql query that retrieves
        the corresponding wikidata URIs;
        3. the match between MovieLens-100k indexes and wikidata URIs contained in MindReader is created;
        4. the wikidata URIs are converted in the corresponding MindReader indexes using the indexing computed
        during pre-processing;
        5. the same procedure is repeated between titles and years of unmatched MovieLens-100k and MindReader movies.

        Returns a dictionary containing this mapping, namely ml_idx -> mr_idx.
        """
        if os.path.exists(os.path.join(self.data_path, "mapping.csv")):
            ml_100_idx_to_mr_idx = pd.read_csv("./datasets/mapping.csv").to_dict("records")
            ml_100_idx_to_mr_idx = {match["ml_idx"]: match["mr_idx"] for match in ml_100_idx_to_mr_idx}
        else:
            # create mapping between movie indexes in ml-latest and iMDB ids
            ml_link_file = pd.read_csv(os.path.join(self.data_path, "ml-latest-small/links.csv"), dtype=str)
            ml_link_file["movieId"] = ml_link_file["movieId"].astype(np.int64)
            ml_link_file = ml_link_file.to_dict("records")
            ml_idx_to_link = {link["movieId"]: "tt" + link["imdbId"] for link in ml_link_file}
            # create mapping between movie titles in ml-latest and their indexes
            # note that there are two "Emma" in MovieLens-latest, but only one is in MindReader. Other Emma has
            # been manually canceled
            ml_movies_file = pd.read_csv(os.path.join(self.data_path, "ml-latest-small/movies.csv")).to_dict("records")
            ml_title_to_idx = {movie["title"]: movie["movieId"] for movie in ml_movies_file}
            # get iMDB ids for movies in MovieLens-100k
            ml_100_title_to_idx = {movie[1]: movie[0] for movie in self.ml_100_movie_info}
            ml_100_idx_to_title = {idx: title for title, idx in ml_100_title_to_idx.items()}
            # match titles between MovieLens-100k and MovieLens-latest movies to get imdb id of MovieLens-100k movies
            ml_100_idx_to_imdb = {}
            matched = []  # it contains matched titles
            for title, idx in ml_100_title_to_idx.items():
                if title in ml_title_to_idx:
                    # if title of 100k is in ml-latest dataset, get its link
                    ml_100_idx_to_imdb[idx] = ml_idx_to_link[ml_title_to_idx[title]]
                    matched.append(title)
            # for remaining unmatched movies, we use fuzzy methods since the titles have some marginal differences
            # (e.g., date, order of words)
            no_matched_100k = set(ml_100_title_to_idx.keys()) - set(matched)
            no_matched_ml = set(ml_title_to_idx.keys()) - set(matched)

            for ml_100_title in no_matched_100k:
                try:
                    # get year of ml-100k movie
                    f_year = int(ml_100_title.strip()[-6:][1:5])
                except ValueError:
                    # if there is no year info, we do not find matches
                    continue
                best_sim = 0.0
                candidate_ml_100_title = ""
                candidate_ml_title = ""
                for ml_title in no_matched_ml:
                    try:
                        # get year of ml-latest movie
                        s_year = int(ml_title.strip()[-6:][1:5])
                    except ValueError:
                        continue
                    # compute similarity between titles with year removed
                    title_sim = fuzz.token_set_ratio(ml_100_title.strip()[:-6], ml_title.strip()[:-6])
                    # search for best similarity between titles
                    if title_sim > best_sim and ml_100_title not in matched and ml_title not in matched \
                            and f_year == s_year:
                        best_sim = title_sim
                        candidate_ml_100_title = ml_100_title
                        candidate_ml_title = ml_title
                if best_sim >= 96:
                    # here, it erroneously matches "My Life and Times With Antonin Artaud
                    # (En compagnie d'Antonin Artaud)" in ml-100k with "My Life" in ml-latest
                    # we manually modify the title of the first movie with just "En compagnie d'Antonin Artaud" in such
                    # a way the procedure is not able to find the match anymore
                    # it seems that under 96 there are typos in the names and the matches are not reliable
                    # add movie to the dict containing the matches
                    ml_100_idx_to_imdb[ml_100_title_to_idx[candidate_ml_100_title]] = \
                        ml_idx_to_link[ml_title_to_idx[candidate_ml_title]]
                    if candidate_ml_100_title == candidate_ml_title:
                        # match only one if they are the same
                        matched.append(candidate_ml_title)
                    else:
                        # match both if they are different since the `matched` list is shared between datasets
                        matched.append(candidate_ml_100_title)
                        matched.append(candidate_ml_title)
            # update not matched 100k titles
            no_matched_100k = list(set(no_matched_100k) - set(matched))

            # now, we need to fetch the wikidata URIs for the movies we have matched between MovieLens-100k and
            # MovieLens-latest-small

            query = "SELECT ?label ?film WHERE {?film wdt:P345 ?label.VALUES ?label {" + \
                    ' '.join(["'" + imdb_url + "'" for idx, imdb_url in ml_100_idx_to_imdb.items()]) + \
                    "} SERVICE wikibase:label { bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en'. } }"

            result = send_query(query)

            ml_100_imdb_to_wikidata_uri = {match['label']['value']: match['film']['value'] for match in result}

            # create mapping from MovieLens-100k indexes to Wikidata URIs - this mapping contains an entry only if the
            # previous query found the link between ml-100k movie iMDB id and Wikidata entity - for 3 movies the
            # link is not found
            ml_100_idx_to_wikidata_uri = {}
            for idx, imdb_id in ml_100_idx_to_imdb.items():
                if imdb_id in ml_100_imdb_to_wikidata_uri:
                    ml_100_idx_to_wikidata_uri[idx] = ml_100_imdb_to_wikidata_uri[imdb_id]
                else:
                    # update no matched 100k movies - we need to update it because we need to repeat a similar procedure
                    # onl for unmatched movies
                    no_matched_100k.append(ml_100_idx_to_title[idx])
                    matched.remove(ml_100_idx_to_title[idx])

            # remove from the mapping all matches in which the wikidata uri does not appear in the MindReader dataset
            # we do not need these matches since we are interested in matching ml-100k with MindReader
            # fetch uri to idx mapping computed by the pre-processing procedure - we need to convert the fetched uris
            # with proper indexes
            mr_uri_to_idx = pd.read_csv(os.path.join(self.data_path,
                                                     "mindreader/processed/item_mapping.csv")).to_dict("records")
            mr_uri_to_idx = {mapping["old_idx"]: mapping["new_idx"] for mapping in mr_uri_to_idx}
            mr_idx_to_uri = {idx: uri for uri, idx in mr_uri_to_idx.items()}
            mr_uri_to_title = {mr_idx_to_uri[movie["idx"]]: movie["title"] for movie in self.mr_movie_info}
            mr_title_to_uri = {title: uri for uri, title in mr_uri_to_title.items()}
            # the match is kept only if the retrieved wikidata URI is in the corpus of movies of MindReader
            ml_100_idx_to_mr_uri = {}
            for idx, uri in ml_100_idx_to_wikidata_uri.items():
                # note that mr_uri_to_title contains uris of movies that have been rated by at least a user
                # not rated movies are not included in the dataset and hence also the matches between not rated movies
                if uri in mr_uri_to_title:
                    ml_100_idx_to_mr_uri[idx] = uri
                else:
                    # update matched and no matched 100k movies
                    no_matched_100k.append(ml_100_idx_to_title[idx])
                    matched.remove(ml_100_idx_to_title[idx])

            # compute no matched MindReader movies
            no_matched_mr = set(mr_uri_to_title.keys()) - set(ml_100_idx_to_mr_uri.values())

            # repeat fuzzy procedure between MovieLens-100k and MindReader directly, since there could be some errors
            # in the links of the ml-latest dataset

            for ml_100_title in no_matched_100k:
                try:
                    # get year of ml-100k movie
                    f_year = int(ml_100_title.strip()[-6:][1:5])
                except ValueError:
                    continue
                best_sim = 0.0
                candidate_ml_100_title = ""
                candidate_mr_title = ""
                for uri in no_matched_mr:
                    mr_title = mr_uri_to_title[uri]
                    try:
                        # get year of MindReader movie
                        s_year = int(mr_title.strip()[-6:][1:5])
                    except ValueError:
                        continue
                    # compute similarity between titles with year removed
                    title_sim = fuzz.token_set_ratio(ml_100_title.strip()[:-6], mr_title.strip()[:-6])
                    # search for best similarity between titles
                    if title_sim > best_sim and ml_100_title not in matched and mr_title not in matched \
                            and f_year == s_year:
                        best_sim = title_sim
                        candidate_ml_100_title = ml_100_title
                        candidate_mr_title = mr_title
                if best_sim >= 96:
                    # here, it erroneously matches "Big Blue, The (Grand bleu, Le)" of ml-100k with "Big" of MindReader
                    # we set the title of the second movie to "Big (reg. Penny Marshall)" in such a way the procedure
                    # does not find the match anymore
                    ml_100_idx_to_mr_uri[ml_100_title_to_idx[candidate_ml_100_title]] = \
                        mr_title_to_uri[candidate_mr_title]
                    if candidate_ml_100_title == candidate_mr_title:
                        matched.append(candidate_ml_100_title)
                    else:
                        matched.append(candidate_ml_100_title)
                        matched.append(candidate_mr_title)

            # convert MindReader uris to proper indexes computed by the pre-processing procedure
            ml_100_idx_to_mr_idx = {idx: mr_uri_to_idx[uri] for idx, uri in ml_100_idx_to_mr_uri.items()}
            # save the mapping for save time in the next run - we compute it once
            mapping = pd.DataFrame.from_dict({"ml_idx": list(ml_100_idx_to_mr_idx.keys()),
                                              "mr_idx": list(ml_100_idx_to_mr_idx.values())})
            mapping.to_csv("./datasets/mapping.csv", index=False)

        return ml_100_idx_to_mr_idx

    def get_ml_union_mr_dataset(self):
        """
        Dataset containing the movie ratings of ml-100k, and the movie ratings and movie genres' ratings of MindReader.
        The joint movies share the same index in the new dataset.

        It creates a new dataset by merging the ratings of ml-100k and MindReader. The match between the movies
        of ml-100k and the movies of MindReader is used to assign a unique index for those movies in the new dataset.
        For the other movies, namely movies which appear only on ml-100k or only on MindReader, a new index is assigned.

        The sets of users of the two datasets are disjoint. For ml-100k, the users rated only movies, while for
        MindReader they rated also movie genres and other entities. In these experiments, we use only the ratings on
        movie genres and not also on other entities since they are less represented.

        Only the most popular genres are taken into account, namely the genres appearing in the MovieLens-100k dataset.
        They are just 18. The ratings on the other genres of MindReader (more than 150 genres) are not considered since
        they are sub-genres of the main genres.

        The idea of this dataset is to test whether the information about the preferences on movie genres helps in
        improving the accuracy on the MovieLens-100k, especially for the movies that do not belong to the intersection
        of the two datasets. For those movies, a knowledge transfer formula could help in transferring knowledge learnt
        in MindReader to ml-100k. It is a kind of knowledge transfer from MindReader to MovieLens-100k.

        We need to pay attention to the fact that it could be the information provided by the movie ratings of
        MindReader to increase the performance. For this reason, an ablation study should be done. We need to train both
        with genres and without genres.

        Notes:
            - the joint movies are mapped to the same new index
            - other movies are mapped to new indexes
            - this indexing is done in such a way that the indexes are in the range [0, n_movies]
            - users from ml-100k and MindReader are mapped to the same index space in such a way that they are in the
            range [0, n_users]
            - the indexes of the genres are in the range [0, 18]. Use the `self.genres` list to understand to which
            genre the index corresponds
            - the function returns also the 4 mappings that have been internally built. They could be useful to come
            back to original datasets from the new one
        """
        # for creating the fusion dataset, we need to take all the movies of ml-100k and all the movies of MindReader
        # since some movies are shared across the datasets, they will share the same idx in the new dataset
        # idea: we create indexes for joint movies between the datasets, then we create indexes for the remaining movies
        # in ml-100k and MindReader
        # create unique indexing for shared movies between datasets
        ml_to_new_idx = {ml_idx: idx for idx, ml_idx in enumerate(self.ml_to_mr.keys())}
        mr_to_new_idx = {mr_idx: idx for idx, mr_idx in enumerate(self.ml_to_mr.values())}
        # create indexing for movies which are not shared
        ml_ratings = np.array([(rating[0], rating[1], rating[2]) for rating in self.ml_100_ratings])
        # create indexes for ml-100k movies not in the joint set
        j = len(self.ml_to_mr)  # we start from the first available index after the joint movies
        for movie in self.ml_100_movie_info:
            if movie[0] not in self.ml_to_mr:
                ml_to_new_idx[movie[0]] = j
                j += 1
        # create indexes for MindReader movies not in the joint set
        mr_movie_ratings = np.array([(rating["u_idx"], rating["i_idx"], rating["rate"])
                                     for rating in self.mr_movie_ratings])
        for movie in self.mr_movie_info:
            # here, we take the values() because we need matched MindReader movies
            if movie["idx"] not in self.ml_to_mr.values():
                mr_to_new_idx[movie["idx"]] = j
                j += 1
        # create unique user indexing
        # here, we have two distinct mappings because in both datasets the indexes of the users start from 0 after
        # the pre-processing procedure
        user_mapping_ml = {}
        user_mapping_mr = {}
        i = 0
        for user in set(ml_ratings[:, 0]):
            user_mapping_ml[user] = i
            i += 1
        for user in set(mr_movie_ratings[:, 0]):
            user_mapping_mr[user] = i
            i += 1
        # create final dataset
        dataset_ratings = [{"u_idx": user_mapping_ml[u], "i_idx": ml_to_new_idx[i], "rate": r}
                           for u, i, r in ml_ratings]
        dataset_ratings.extend([{"u_idx": user_mapping_mr[u], "i_idx": mr_to_new_idx[i], "rate": r}
                                for u, i, r in mr_movie_ratings])
        dataset_ratings = pd.DataFrame.from_records(dataset_ratings)

        # we need to associate each movie to its genres
        # in the union, we need to use also the genres of MovieLens-100k, since we have some movies for which we do not
        # have the information in MindReader
        # we use only the most common genres, to reduce sparsity (genres of MovieLens-100k)
        # for the joint movies, we use the genres in MindReader, since they are more accurate and updated
        # take the genres of all movies in MindReader first
        # todo da notare che in tutte queste strutture si utilizzano gli indici finali, verificare che non uso un
        #  mapping per convertirli, perche' non ce ne' bisogno
        dataset_movie_to_genres = {mr_to_new_idx[movie["idx"]]: movie["genres"].split("|")
                                   for movie in self.mr_movie_info}
        # take the genres for remaining movies
        dataset_movie_to_genres = {**dataset_movie_to_genres, **{ml_to_new_idx[movie[0]]: movie[2].split("|")
                                                                 for movie in self.ml_100_movie_info
                                                                 if ml_to_new_idx[movie[0]] not in dataset_movie_to_genres}}
        # create np.array of movie-genre pairs
        dataset_movie_to_genres = np.array([[movie, int(genre)] for movie in dataset_movie_to_genres
                                            for genre in dataset_movie_to_genres[movie]
                                            if genre != 'None'])
        # convert the array in a sparse matrix (items X genres)
        movie_genres_matrix = csr_matrix((np.ones(len(dataset_movie_to_genres)),
                                          (dataset_movie_to_genres[:, 0], dataset_movie_to_genres[:, 1])),
                                         shape=(j, len(self.genres)))
        # now, we need to associate each user to the genres she likes or dislikes - this info is only on MindReader
        # get ratings of users of MindReader for the genres
        mr_genre_ratings = [(rating["u_idx"], rating["g_idx"], rating["rate"])
                            for rating in self.mr_genre_ratings]

        # this dict contains the genres that each user likes and the genres that each user dislikes
        # dataset_user_to_genres = {}
        # get new index for users of MindReader that only rated movie genres
        for u, g, r in mr_genre_ratings:
            if u not in user_mapping_mr:  # some users have only rated movie genres, so we need to add them to the map
                user_mapping_mr[u] = i
                i += 1
            # if user_mapping_mr[u] not in dataset_user_to_genres:
            #     dataset_user_to_genres[user_mapping_mr[u]] = {"likes": [g], "dislikes": []} \
            #         if r == 1 else {"likes": [], "dislikes": [g]}
            # else:
            #     if r == 1:
            #         dataset_user_to_genres[user_mapping_mr[u]]["likes"].append(g)
            #     else:
            #         dataset_user_to_genres[user_mapping_mr[u]]["dislikes"].append(g)
        # get genre_ratings as numpy array by also mapping the users to the new indexes of the final dataset
        mr_genre_ratings = np.array([(user_mapping_mr[u], g, r) for u, g, r in mr_genre_ratings])

        # unify user and item mappings
        idx_mapping = {"user": {"ml": user_mapping_ml, "mr": user_mapping_mr},
                       "item": {"ml": ml_to_new_idx, "mr": mr_to_new_idx}}

        return dataset_ratings, movie_genres_matrix, mr_genre_ratings, idx_mapping

    def get_ml_union_mr_genre_ratings_dataset(self):
        """
        Dataset containing the movie ratings of ml-100k and the movie genres' ratings of MindReader.

        The idea of this dataset is to test whether the information about the genres in MindReader (just this info)
        helps in improving the accuracy on the MovieLens-100k. The only way to transfer information from the genre
        ratings of MindReader to ml-100k is to use a knowledge transfer formula in LTN.

        In practice, without knowledge transfer, the information can be transfer just thanks to the loss aggregation.
        In the same batch, we will have ratings on movies of ml-100k and ratings on genres of MindReader. When the
        predictions are aggregated, some information is shared across genres and movies since the overall signal is used
        to update the weights.

        By using a knowledge transfer formula, it could be possible to improve performances on ml-100k and to deal
        with cold-start issues. In the cold-start cases, we have a few number of ratings. We can use the knowledge
        transfer formula to predict genres preferences and then use this information to predict the best movies to
        recommend.
        """
        # take ml-100k ratings
        ml_ratings = np.array([(rating[0], rating[1], rating[2]) for rating in self.ml_100_ratings])
        # we need to associate each movie to its genres
        # we use the information in MindReader for the movies in the joint set and the information of ml-100k for the
        # others
        # we construct the inverse indexing to get genres of MindReader for movies in ml-100k that are in the joint set
        mr_to_ml = {mr_idx: ml_idx for ml_idx, mr_idx in self.ml_to_mr.items()}
        # get genres of MindReader for ml-100k movies in the joint set
        dataset_movie_to_genres = {mr_to_ml[movie["idx"]]: movie["genres"].split("|")
                                   for movie in self.mr_movie_info if movie["idx"] in mr_to_ml}
        # get genres for remaining movies in ml-100k
        dataset_movie_to_genres = {**dataset_movie_to_genres, **{movie[0]: movie[2].split("|")
                                                                 for movie in self.ml_100_movie_info
                                                                 if movie[0] not in dataset_movie_to_genres}}
        # create numpy array of movie-genre pairs
        dataset_movie_to_genres = np.array([[movie, int(genre)] for movie in dataset_movie_to_genres
                                            for genre in dataset_movie_to_genres[movie]
                                            if genre != 'None'])
        # convert array to sparse matrix
        movie_genres_matrix = csr_matrix((np.ones(len(dataset_movie_to_genres)),
                                         (dataset_movie_to_genres[:, 0], dataset_movie_to_genres[:, 1])),
                                         shape=(len(np.unique(ml_ratings[:, 1])), len(self.genres)))
        # get genre ratings in MindReader
        mr_genre_ratings = np.array([(rating["u_idx"], rating["g_idx"], rating["rate"])
                                     for rating in self.mr_genre_ratings])
        # take the first available user index after the users of ml-100k - we can do this because in the training set
        # there is at least one rating for each user
        u = len(set(ml_ratings[:, 0]))
        # map mr users to new indexes - we need to do this because both ml-100k user indexes and MindReader user
        # indexes start from 0
        user_mapping_mr = {}
        for user in set(mr_genre_ratings[:, 0]):
            if user not in user_mapping_mr:
                user_mapping_mr[user] = u
                u += 1
        # map users to the new correct index
        # this is done because both users in ml-100k and MindReader have indexes starting from 0
        # we have to avoid to map different users to the same index
        mr_genre_ratings = np.array([[user_mapping_mr[u], g, r] for u, g, r in mr_genre_ratings])
        # the code that follows is used just because crate_fusion_folds needs this info
        # there is not a new indexing for ml-100k users and movies
        # item_mapping_ml = {movie_idx: movie_idx for movie_idx in set(ml_ratings[:, 1])}
        # user_mapping_ml = {user_idx: user_idx for user_idx in set(ml_ratings[:, 0])}
        # we need the ratings in the records format since create_fusion_folds requires this format
        ml_ratings = pd.DataFrame.from_records([{"u_idx": u, "i_idx": i, "rate": r} for u, i, r in ml_ratings])
        # unify index mappings

        return ml_ratings, movie_genres_matrix, mr_genre_ratings

    def get_ml100k_folds(self, seed, mode='ml', fair=True, n_neg=100):
        """
        Creates and returns the training, validation, and test set of the MovieLens-100k dataset.

        The procedure starts from the entire datasets. To create the test set, one random positive movie rating for
        each user is held-out. To complete the test example of the user, `n_neg` irrelevant movies (movies never seen
        by the user in the dataset) are randomly sampled.
        The metrics evaluate the recommendation based on the position of the positive movie in a ranking containing all
        the `n_neg` + 1 movies.

        The validation set is constructed in the same way, starting from the remaining positive ratings in the dataset.

        Note that the test positive rating is held-out only if the user has at least 2 ratings, in such a way that at
        least one positive rating remains in the training set. The same idea is applied to the validation set.

        The `seed` parameter is used for reproducibility of experiments (random sample of positive and negative movies).

        The `mode` parameter is used to control the selection of candidate test movies:
            - If mode is 'ml', a random positive movie for each user is held out for validation and test from the set of
            ml-100k ratings. Note that in this evaluation type, the target item could belong to the joint movies or to
            the movies appearing only in the ml-100k dataset.
            - If mode is equal to "ml\\mr", the candidate test movies are randomly sampled from the set of movies that
            appear only in the ml-100k dataset and not also in the joint set of movies. This mode is used to see how the
            knowledge transfer technique works in challenging scenarios. Intuitively, ml\\mr means the difference
            between the set of ratings in ml-100k and the set of ratings in MindReader, namely, the ratings on movies
            that appear only in ml-100k.
            - If mode is equal to "ml&mr", the candidate test movies are randomly sampled for the joint set of movies.
            Higher scores are expected for these movies since the intersection is where the knowledge is easily
            transferred between the two datasets. In fact, a movie in the joint set could have ratings from both users
            in ml-100k and users in MindReader.

        The `n_neg` parameter is used to decide the number of random negative samples used to construct each
        validation/test example. The higher the number the more challenging will be for the model to put
        the target item in the top-N positions of the ranking.

        The `fair` parameter is used when 'ml\\mr' or 'ml&mr' are used. If fair is set to True, then the 'ml\\mr' will
        produce negative items which are only present in the ml-100k dataset and not in MindReader. If it is False, it
        will produce negative items that could be also in the joint set of movies. The same is valid for 'ml&mr'.
        If fair is True, the negative items will be sampled from the joint movies, if it is False, from both ml-100k
        and the joint set movies.
        """
        assert mode == "ml" or mode == "ml\mr" or mode == "ml&mr", "The selected mode (%s) does not exist" % (mode, )
        # set seed for the sampling of random negative items
        set_seed(seed)
        # we need a dataframe because it allows to apply group by operations
        ratings = pd.DataFrame.from_records(self.ml_100_ratings)
        n_users = ratings[0].nunique()
        n_items = ratings[1].nunique()
        # group by user idx
        groups = ratings.groupby(by=[0])
        # get set of movie indexes
        ml_movies = set(ratings[1].unique())
        # get movies in ml\mr
        only_ml_movies = ml_movies - set(self.ml_to_mr.keys())
        validation = []
        test = []
        for u_idx, u_ratings in groups:
            # take n_neg randomly sampled negative (not seen by the user) movies for test and validation
            not_seen_by_u = ml_movies - set(u_ratings[1])
            # positive ratings for the user
            pos_u_ratings = u_ratings[u_ratings[2] == 1]
            # filter pos_u_ratings based on selected mode
            if mode == "ml\mr":
                # remove from pos_u_ratings all the positive ratings which belong to the intersection dataset
                # (we want the movies that are only in ml-100k)
                pos_u_ratings = pos_u_ratings[~pos_u_ratings[1].isin(self.ml_to_mr)]
                if fair:
                    # remove from negative movies the movies in the joint set
                    not_seen_by_u -= set(self.ml_to_mr.keys())
            if mode == "ml&mr":
                # remove from pos_u_ratings all the positive ratings which do not belong to the intersection
                # dataset (just fusion movies)
                pos_u_ratings = pos_u_ratings[pos_u_ratings[1].isin(self.ml_to_mr)]
                if fair:
                    # remove from negative movies the movies which are not in the joint set
                    not_seen_by_u -= only_ml_movies
            # check if it is possible to sample - leave at least one positive rating in the training set
            # if there is only one positive ratings for user u, we leave that rating in the training set
            # the validation and test set will not include this specific user
            if len(pos_u_ratings) > 1:
                # if we have at least three positive ratings, take one random positive rating for test and one
                # for validation
                if len(pos_u_ratings) > 2:
                    sampled_pos_u_ratings = pos_u_ratings.sample(n=2, random_state=seed)
                else:
                    # if we have just two ratings, we held-out one for test and leave the other one in the training set
                    sampled_pos_u_ratings = pos_u_ratings.sample(n=1, random_state=seed)
                # remove sampled ratings from the dataset, since they are ratings held out for validation and/or test
                ratings.drop(sampled_pos_u_ratings.index, inplace=True)
                # get list of sampled item indexes
                sampled_pos_u_ratings = list(sampled_pos_u_ratings[1])
                # we cannot increase n_neg too much since we do not have a lot of negative movies when the mode is
                # set to ml&mr
                assert len(not_seen_by_u) > n_neg, "Problem: parameter n_neg (%d) is higher than the available " \
                                                   "number of movies from which we can sample (%d). Set a lower " \
                                                   "number of n_neg" % (n_neg, len(not_seen_by_u))
                neg_movies_test_for_u = random.sample(not_seen_by_u, n_neg)
                test.append([[u_idx, item] for item in neg_movies_test_for_u + [sampled_pos_u_ratings[0]]])
                # if we have at least three positive ratings, it means we have have to create also the validation
                # example for u
                if len(pos_u_ratings) > 2:
                    neg_movies_val_for_u = random.sample(not_seen_by_u, n_neg)
                    validation.append([[u_idx, item] for item in neg_movies_val_for_u + [sampled_pos_u_ratings[1]]])

        validation = np.array(validation)
        test = np.array(test)

        # create np array of training ratings
        train = ratings.to_dict("records")
        train = np.array([(rating[0], rating[1], rating[2]) for rating in train])

        return Dataset(train, validation, test, n_users, n_items, name="ml")

    def get_mr_folds(self, seed, n_neg=100, with_genres=False):
        """
        Creates and returns the training, validation, and test set of the MindReader dataset.

        The procedure starts from the entire datasets. To create the test set, one random positive movie rating for
        each user is held-out. To complete the test example of the user, `n_neg` irrelevant movies (movies never seen
        by the user in the dataset) are randomly sampled.
        The metrics evaluate the recommendation based on the position of the positive movie in a ranking containing all
        the `n_neg` + 1 movies.

        The validation set is constructed in the same way, starting from the remaining positive ratings in the dataset.

        Note that the test positive rating is held-out only if the user has at least 2 ratings, in such a way that at
        least one positive rating remains in the training set. The same idea is applied to the validation set.

        The `seed` parameter is used for reproducibility of experiments (random sample of positive and negative movies).

        The `n_neg` parameter is used to decide the number of random negative samples used to construct each
        validation/test example. The higher the number the more challenging will be for the model to put
        the target item in the top-N positions of the ranking.

        The `with_genres` parameter is used to decide whether to include the ratings for movie genres in the
        training set. By default, it is `False`.
        """
        # set seed for the sampling of random negative items
        set_seed(seed)
        # we need a dataframe because it allows to apply group by operations
        ratings = pd.DataFrame.from_records(self.mr_movie_ratings)
        # get user ids - this is used after if the user wants to add the ratings for the genres to the dataset
        user_ids = ratings["u_idx"].unique()
        # get number of users and items
        n_users = ratings["u_idx"].nunique()
        n_items = ratings["i_idx"].nunique()
        # group by user idx
        groups = ratings.groupby(by=["u_idx"])
        # take set of mr movies - used to compute the set of negative movies for each user
        mr_movies = set(ratings["i_idx"].unique())
        # initialize lists that contain validation and test examples
        validation, test = [], []
        for u_idx, u_ratings in groups:
            # compute the set of negative movies for the current user
            not_seen_by_u = mr_movies - set(u_ratings["i_idx"])
            # positive ratings for the user
            pos_u_ratings = u_ratings[u_ratings["rate"] == 1]
            # check if it is possible to sample - leave at least one positive rating in the training set
            # if there is only one positive ratings for user u, we leave that rating in the training set
            # the validation and test set will not include this specific user
            if len(pos_u_ratings) > 1:
                # if we have at least three positive ratings, take one random positive rating for test and one
                # for validation
                if len(pos_u_ratings) > 2:
                    sampled_pos_u_ratings = pos_u_ratings.sample(n=2, random_state=seed)
                else:
                    # if we have just two ratings, we held-out one for test and leave the other one in the training set
                    sampled_pos_u_ratings = pos_u_ratings.sample(n=1, random_state=seed)
                # remove sampled ratings from the dataset, since they are ratings held out for validation and/or test
                ratings.drop(sampled_pos_u_ratings.index, inplace=True)
                # get list of sampled item indexes
                sampled_pos_u_ratings = list(sampled_pos_u_ratings["i_idx"])
                # check if we have enough movie to sample
                assert len(not_seen_by_u) > n_neg, "Problem: parameter n_neg (%d) is higher than the available " \
                                                   "number of movies from which we can sample (%d). Set a lower " \
                                                   "number of n_neg" % (n_neg, len(not_seen_by_u))
                neg_movies_test_for_u = random.sample(not_seen_by_u, n_neg)
                test.append([[u_idx, item] for item in neg_movies_test_for_u + [sampled_pos_u_ratings[0]]])
                # if we have at least three positive ratings, it means we have have to create also the validation
                # example for u
                if len(pos_u_ratings) > 2:
                    neg_movies_val_for_u = random.sample(not_seen_by_u, n_neg)
                    validation.append([[u_idx, item] for item in neg_movies_val_for_u + [sampled_pos_u_ratings[1]]])

        validation = np.array(validation)
        test = np.array(test)

        # create np array of training ratings
        train = ratings.to_dict("records")
        train = np.array([(rating["u_idx"], rating["i_idx"], rating["rate"]) for rating in train])

        # add ratings for movie genres if it has been requested by the user
        if with_genres:
            # compute itemXgenres matrix
            # get genres of MindReader movies
            movie_to_genres = {movie["idx"]: movie["genres"].split("|") for movie in self.mr_movie_info}
            # create np.array of movie-genre pairs
            movie_to_genres = np.array([[movie, int(genre)] for movie in movie_to_genres
                                        for genre in movie_to_genres[movie]
                                        if genre != 'None'])
            # convert the array in a sparse matrix (items X genres)
            movie_genres_matrix = csr_matrix((np.ones(len(movie_to_genres)),
                                              (movie_to_genres[:, 0], movie_to_genres[:, 1])),
                                             shape=(n_items, len(self.genres)))
            # get ratings for movie genres
            genre_ratings = pd.DataFrame.from_records(self.mr_genre_ratings)
            genre_ratings = np.array(genre_ratings)
            # update the number of users - some users have only rated movie genres
            n_users = len(set(user_ids) | set(genre_ratings[:, 0]))
            n_movies = n_items
            # update the number of items - we have added the movie genres to the dataset
            n_items = n_items + len(set(genre_ratings[:, 1]))  # we consider genres as items
            # move the genre indexes after the movie indexes in the dataset -> the genre embeddings will be the last
            # in the embedding tensor of the model
            new_genre_ratings = np.array([[u, g + n_movies, r] for u, g, r, in genre_ratings])
            # the training set contains both ratings on movies and on movie genres
            return DatasetWithGenres(np.concatenate([train, new_genre_ratings], axis=0),
                                     validation, test, n_users, n_items,
                                     name="mr(movies+genres)", n_genres=len(self.genres),
                                     n_genre_ratings=len(new_genre_ratings), item_genres_matrix=movie_genres_matrix)

        return Dataset(train, validation, test, n_users, n_items, name="mr(movies)")

    def get_fusion_folds_given_ml_folds(self, train_set, ml_val, ml_test, idx_mapping=None, genre_ratings=None,
                                        item_genres_matrix=None):
        """
        It constructs the train, validation, and test set for the dataset which is the fusion between ml-100k and
        MindReader datasets.
        It takes as input the validation set and test set of ml-100k in such a way to find the same ratings in the
        fusion dataset. We want the validation and test sets among the datasets to be identical since we need to
        test whether the fusion dataset provides improvements w.r.t. ml-100k alone.

        The movie and user mapping are required to find the same triples in the fusion dataset, since it uses a
        different indexing w.r.t. to ml-100k.

        If parameter `genre_ratings` is different from None (np.array of genre ratings is given), the function returns
        also a np.array containing the ratings for the genres. Take into account that since some users have only
        rated genres, the number of users between the two configurations might be different.

        If parameter `genre_ratings` is not None, `item_genres_matrix` should not be None. It is an itemsXgenres sparse
        matrix containing the genres of each movie.
        """
        # todo e' qui che dovrei lasciare dei generi in test se voglio fare la verifica
        fusion_val = ml_val
        fusion_test = ml_test
        # if idx_mapping is not None, it means that the new dataset has a different indexing, so we need to find the
        # correct user-item pair that need to be removed from the training set and put into validation and test sets
        # the correct user-item pairs are into ml_val and ml_test. We use the idx_mapping to find them in the new
        # dataset
        # if idx_mapping is None, it means that the new dataset has the same indexing of ml_val and ml_test, so they
        # can be directly used
        if idx_mapping is not None:
            # find the same validation examples in the fusion dataset
            fusion_val = np.array([[[idx_mapping["user"]["ml"][u], idx_mapping["item"]["ml"][i]] for u, i in user]
                                   for user in ml_val])
            # find the same test examples in the fusion dataset
            fusion_test = np.array([[[idx_mapping["user"]["ml"][u], idx_mapping["item"]["ml"][i]] for u, i in user]
                                    for user in ml_test])
        # create training set by removing the validation/test target ratings
        to_remove = fusion_val[:, -1]
        to_remove = np.concatenate((to_remove, fusion_test[:, -1]), axis=0).tolist()
        train_set_dict = train_set.to_dict("records")
        fusion_train = np.array([[rating["u_idx"], rating["i_idx"], rating["rate"]]
                                 for rating in train_set_dict if [rating["u_idx"], rating["i_idx"]] not in to_remove])
        if genre_ratings is not None:
            assert item_genres_matrix is not None, "Parameter item_genres_matrix is None even if genre_ratings is " \
                                                   "not None. Please, pass a itemXgenres matrix."
            n_users = len(set(train_set["u_idx"].unique()) | set(genre_ratings[:, 0]))
            n_items = train_set["i_idx"].nunique() + len(set(genre_ratings[:, 1]))  # we consider genres as items
            n_movies = train_set["i_idx"].nunique()
            # move the genre indexes after the movie indexes in the dataset -> the genre embeddings will be the last
            # in the embedding tensor of the model
            new_genre_ratings = np.array([[u, g + n_movies, r] for u, g, r, in genre_ratings])
            # the training set contains both ratings on movies and on movie genres
            return DatasetWithGenres(np.concatenate([fusion_train, new_genre_ratings], axis=0),
                                     fusion_val, fusion_test, n_users, n_items,
                                     name="ml|mr(movies+genres)" if idx_mapping is not None else "ml(movies)|mr(genres)",
                                     n_genres=len(self.genres), n_genre_ratings=len(new_genre_ratings),
                                     item_genres_matrix=item_genres_matrix)

        return Dataset(fusion_train, fusion_val, fusion_test, train_set["u_idx"].nunique(),
                       train_set["i_idx"].nunique(), name="ml|mr(movies)")

    @staticmethod
    def increase_data_sparsity(dataset, p_to_keep, seed):
        """
        It removes ratings from the training set of the given dataset to increase its sparsity.
        If the dataset includes some genre ratings, it only removes movie ratings.

        :param dataset: the Dataset object from which we need to remove training ratings
        :param p_to_keep: the proportion of ratings that has to be kept for each user of the training set. The other
        ratings will be discarded
        :param seed: seed for reproducibility of the experiments
        :return the dataset with a proportion of training ratings per user equal to `p_to_keep`, without removing genre
        ratings if there are
        """
        assert 0 < p_to_keep <= 1, "The proportion of ratings to be kept for each user is out of the possible range."
        # if p_to_keep is 1, it means that we do not have to remove anything
        if p_to_keep != 1:
            genre_ratings = None  # initialization
            train_df = pd.DataFrame(dataset.train, columns=["uid", "iid", "rate"])
            if isinstance(dataset, DatasetWithGenres):
                # remove genre ratings to avoid removing them with the random sampling
                genre_ratings = train_df[-dataset.n_genre_ratings:]
                train_df = train_df[:-dataset.n_genre_ratings]
            groups = train_df.groupby(by=["uid"])
            for u_idx, u_ratings in groups:
                ratings_to_remove = u_ratings.sample(frac=(1 - p_to_keep), random_state=seed)
                train_df.drop(ratings_to_remove.index, inplace=True)

            if isinstance(dataset, DatasetWithGenres):
                dataset.train = np.concatenate([train_df.to_numpy(), genre_ratings], axis=0)
            else:
                dataset.train = train_df.to_numpy()
        return dataset
