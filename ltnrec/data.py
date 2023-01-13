import pandas as pd
import os
import numpy as np
from fuzzywuzzy import fuzz
import random
from SPARQLWrapper import SPARQLWrapper, JSON, POST
from scipy.sparse import csr_matrix
from ltnrec.utils import set_seed


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
    def __init__(self, train, val, test, n_users, n_items, name, n_genres, item_genres_matrix=None):
        super(DatasetWithGenres, self).__init__(train, val, test, n_users, n_items, name)
        self.n_genres = n_genres
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
        dataset_movie_to_genres = dataset_movie_to_genres | {movie[0]: movie[2].split("|")
                                                             for movie in self.ml_100_movie_info
                                                             if movie[0] not in dataset_movie_to_genres}
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
        # set seed for the sampling of random negative items
        set_seed(seed)
        assert mode == "ml" or mode == "ml\mr" or mode == "ml&mr", "The selected mode (%s) does not exist" % (mode,)
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
                # todo forse qua c'e' un bug perche' manca il random state - manca il set del seed - messo su
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
                                     n_genres=len(self.genres), item_genres_matrix=item_genres_matrix)

        return Dataset(fusion_train, fusion_val, fusion_test, train_set["u_idx"].nunique(),
                       train_set["i_idx"].nunique(), name="ml|mr(movies)")

    @staticmethod
    def increase_data_sparsity(dataset, p_to_keep, seed):
        """
        It removes ratings from the training set of the given dataset to increase its sparsity.

        :param dataset: the Dataset object from which we need to remove training ratings
        :param p_to_keep: the proportion of ratings that has to be kept for each user of the training set. The other
        ratings will be discarded
        :param seed: seed for reproducibility of the experiments
        :return the dataset with a proportion of training ratings per user equal to `p_to_keep`
        """
        assert 0 < p_to_keep <= 1, "The proportion of ratings to be kept for each user is out of the possible range."
        if p_to_keep != 1:
            train_df = pd.DataFrame(dataset.train, columns=["uid", "iid", "rate"])
            groups = train_df.groupby(by=["uid"])
            for u_idx, u_ratings in groups:
                ratings_to_remove = u_ratings.sample(frac=(1 - p_to_keep), random_state=seed)
                train_df.drop(ratings_to_remove.index, inplace=True)

            dataset.train = train_df.to_numpy()
        return dataset
