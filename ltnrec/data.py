import copy
import json
import pandas as pd
import os
import numpy as np
import sys
import string
import unidecode
from fuzzywuzzy import fuzz
from SPARQLWrapper import SPARQLWrapper, JSON, POST


class MovieLensMR:
    """
    Class that manages the dataset. The dataset is a union of the MovieLens-100k dataset and the MindReader dataset.
    It is possible to create it in two different ways:
    1. a match between the movies in MovieLens and the movies in MindReader is created and the union of the datasets is
    taken. In this case, there will be 4,940 movies in the dataset;
    2. a match between the movies in MovieLens and the movies in MindReader is created and the intersection of the
    datasets is taken. In this case, there will be 1,682 movies in the dataset.

    The procedure unifies the indexes of the two datasets in the same range.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.check()
        self.idx_to_uri = self.create_mapping()
        self.create_intersection()

    def check(self):
        """
        Check the presence of all needed files:
        1. entities: contains the entities (movies, genres, actors, ...) of the MindReader KG;
        2. ratings: contains the user-entity ratings of the MindReader dataset;
        3. triples: contains the relationships between the entities of the MindReader KG. For example, who is the actor
        of the movie? Which is the genre? Which is the year? All this information is included in the triples file;
        4. u.data (movielens small): contains the user-item ratings of the MovieLens-100k dataset;
        5. u.item (movielens small): contains the content information of the movies in MovieLens-100k;
        6. ratings (movielens big): contains the user-item ratings of the MovieLens latest dataset;
        7. movies (movielens big): contains the content information of the movies in the MovieLens latest dataset;
        8. links (movielens big): contains the links to iMDB for the movies in the dataset
        """
        assert os.path.exists(os.path.join(self.data_path, 'mindreader/entities.csv')), "entities.csv for MindReader " \
                                                                                        "is missing"
        assert os.path.exists(os.path.join(self.data_path, 'mindreader/ratings.csv')), "ratings.csv for MindReader " \
                                                                                       "is missing"
        assert os.path.exists(os.path.join(self.data_path, 'mindreader/triples.csv')), "triples.csv for MindReader " \
                                                                                       "is missing"
        assert os.path.exists(os.path.join(self.data_path, 'ml-100k/u.data')), "u.data for ml-100k is missing"
        assert os.path.exists(os.path.join(self.data_path, 'ml-100k/u.item')), "u.item for ml-100k is missing"
        assert os.path.exists(os.path.join(self.data_path, 'ml-latest-small/ratings.csv')), "ratings.csv for " \
                                                                                            "movielens " \
                                                                                            "latest is missing"
        assert os.path.exists(os.path.join(self.data_path, 'ml-latest-small/movies.csv')), "movies.csv for " \
                                                                                           "movielens " \
                                                                                           "latest is missing"
        assert os.path.exists(os.path.join(self.data_path, 'ml-latest-small/links.csv')), "links.csv for " \
                                                                                          "movielens " \
                                                                                          "latest is missing"

    def create_mapping(self):
        """
        It creates the mapping between the indexes of MovieLens-100k and the URIs of MindReader.

        Returns a dictionary containing this mapping, namely ml_idx -> mr_uri.
        """
        if os.path.exists("./datasets/mapping.csv"):
            ml_100_idx_to_wikidata = pd.read_csv("./datasets/mapping.csv").to_dict("records")
            ml_100_idx_to_wikidata = {mapping["idx"]: mapping["uri"] for mapping in ml_100_idx_to_wikidata}
        else:
            # create index to link dict
            ml_link_file = pd.read_csv(os.path.join(self.data_path, "ml-latest-small/links.csv"), dtype=str)
            ml_link_file["movieId"] = ml_link_file["movieId"].astype(np.int64)
            ml_link_file = ml_link_file.to_dict("records")
            ml_idx_to_link = {link["movieId"]: "tt" + link["imdbId"] for link in ml_link_file}

            # create title to idx dict
            # for movies with multiple indexes the value is a tuple containing the two indexes -> we have no more of these
            # they have been canceled because they were not in the mindreader dataset, so we did not care about them
            # only Emma movie was in MovieLens
            ml_movies_file = pd.read_csv(os.path.join(self.data_path, "ml-latest-small/movies.csv")).to_dict("records")
            ml_title_to_idx = {movie["title"]: movie["movieId"] for movie in ml_movies_file}

            # get imdb id for movies in MovieLens-100k
            ml_100_movies_file = pd.read_csv(os.path.join(self.data_path, "ml-100k/u.item"), sep="|",
                                             encoding="iso-8859-1", header=None).to_dict("records")
            # match titles between MovieLens-100k and MovieLens-latest movies to get imdb id of MovieLens-100k movies
            no_matched_100k = []
            matched = []
            ml_100_idx_to_imdb = {}
            for movie in ml_100_movies_file:
                if movie[1] in ml_title_to_idx:  # if title of 100k is in latest dataset
                    # here, there could be multiple 100k ids pointing the same imdb id
                    # this is ok because movielens-100k has multiple ids for the same movie - this happens if the same user
                    # gives different ratings for the same movie
                    matched.append(movie[1])
                    ml_100_idx_to_imdb[movie[0]] = ml_idx_to_link[ml_title_to_idx[movie[1]]]
                else:
                    no_matched_100k.append(movie)

            # for the remaining unmatched movies I could use fuzzy methods since the titles have some marginal differences
            no_matched_ml_big = set(ml_title_to_idx.keys()) - set(matched)

            for movie_100 in no_matched_100k:
                best_sim = 0.0
                candidate_title_100 = ""
                candidate_title = ""
                candidate_idx = -1
                for movie in no_matched_ml_big:
                    title_sim = fuzz.token_set_ratio(movie_100[1][:-6], movie[:-6])
                    # the year could be slightly different between the two datasets - for this reason I use the
                    # difference of the years
                    try:
                        # in some cases there is not year -> we skip those cases - they are cases in which the cast is
                        # not possible
                        f_year = int(movie_100[1].strip()[-6:][1:5])
                        s_year = int(movie.strip()[-6:][1:5])
                    except:
                        continue
                    if title_sim > best_sim and movie_100[1] not in matched and movie not in matched \
                            and abs(f_year - s_year) <= 1:
                        best_sim = title_sim
                        candidate_idx = movie_100[0]
                        candidate_title_100 = movie_100[1]
                        candidate_title = movie
                if best_sim >= 96:  # it seems that under 96 there are typos in the names and the matches are not reliable
                    # add movie to the dict containing the matches
                    ml_100_idx_to_imdb[candidate_idx] = ml_idx_to_link[ml_title_to_idx[candidate_title]]
                    matched.append(candidate_title_100)
                    matched.append(candidate_title)

            # now, we need to fetch the wikidata URLs for the movies we have matched between MovieLens-100k and
            # MovieLens-latest-small

            query = "SELECT ?label ?film WHERE {?film wdt:P345 ?label.VALUES ?label {" + \
                    ' '.join(["'" + imdb_url + "'" for idx, imdb_url in ml_100_idx_to_imdb.items()]) + \
                    "} SERVICE wikibase:label { bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en'. } }"

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

            imdb_to_wikidata = {match['label']['value']: match['film']['value'] for match in result}

            ml_100_idx_to_wikidata = {idx: imdb_to_wikidata[imdb_id] for idx, imdb_id in ml_100_idx_to_imdb.items()
                                      if imdb_id in imdb_to_wikidata}

            mapping = pd.DataFrame.from_dict({"idx": list(ml_100_idx_to_wikidata.keys()),
                                              "uri": list(ml_100_idx_to_wikidata.values())})
            mapping.to_csv("./datasets/mapping.csv", index=False)

        return ml_100_idx_to_wikidata

    def create_intersection(self):
        # devo unificare i generi dei film su quelli di movielens, che sono quelli piu' comuni
        # su mindreader ci sono troppi generi
        # mi serve un dizionario id to genre
        # per prima cosa devo unificare gli indici, ovvero creare un nuovo mapping sia per movielens che per mindreader
        # questi due mapping devono puntare al mapping unificato
        mr_entities = pd.read_csv(os.path.join(self.data_path, "mindreader/entities.csv")).to_dict("records")
        mr_movies = [entity["uri"] for entity in mr_entities if "Movie" in entity["labels"].split("|")]
        joint_movies = {idx: uri for idx, uri in self.idx_to_uri.items() if uri in mr_movies}
        ml_to_idx = {idx: i for i, idx in enumerate(joint_movies.keys())}
        mr_to_idx = {uri: i for i, uri in enumerate(joint_movies.values())}
        # create inverse mapping too
        idx_to_ml = {idx: ml_idx for ml_idx, idx in ml_to_idx.items()}
        idx_to_mr = {idx: uri for uri, idx in mr_to_idx.items()}
        # take MovieLens-100k ratings
        ml_ratings_file = pd.read_csv(os.path.join(self.data_path, "ml-100k/u.data"),
                                      sep="\t", header=None).to_dict("records")
        ml_ratings = np.array([(rating[0], rating[1], 1 if rating[2] >= 4 else 0) for rating in ml_ratings_file
                               if rating[1] in joint_movies.keys()])
        # take MindReader ratings
        mr_ratings_file = pd.read_csv(os.path.join(self.data_path, "mindreader/ratings.csv")).to_dict("records")
        mr_ratings = [(rating["userId"], rating["uri"], rating["sentiment"] if rating["sentiment"] != -1 else 0)
                      for rating in mr_ratings_file
                      if rating["uri"] in joint_movies.values() if rating["sentiment"] != 0]
        # create unique user indexing
        user_mapping = {}
        i = 0
        for user in ml_ratings[:, 0]:
            if str(user) not in user_mapping:
                user_mapping[str(user)] = i
                i += 1

        for rating in mr_ratings:
            if rating[0] not in user_mapping:
                user_mapping[rating[0]] = i
                i += 1

        # create final dataset
        dataset_ratings = [(user_mapping[str(rating[0])], ml_to_idx[rating[1]], rating[2]) for rating in ml_ratings]
        dataset_ratings.extend([(user_mapping[str(rating[0])], mr_to_idx[rating[1]], rating[2])
                                for rating in mr_ratings])
        dataset_ratings = np.array(dataset_ratings)

        # we need to associate each movie to its genres
        # in this case we can use the genres of MindReader since this is the intersection dataset
        # in the union we need to use also the genres of MovieLens-100k
        # we use only the most common genres, to reduce sparsity (genres of MovieLens-100k)

        genres = ("Action Film", "Adventure Film", "Animated Film", "Children's Film", "Comedy Film", "Crime Film",
                  "Documentary Film", "Drama Film", "Fantasy Film", "Film Noir", "Horror Film", "Musical Film",
                  "Mystery Film", "Romance Film", "Science Fiction Film", "Thriller Film", "War Film", "Western Film")

        mr_triples = pd.read_csv(os.path.join(self.data_path, "mindreader/triples.csv")).to_dict("records")
        mr_uri_to_genre = {entity["uri"]: entity["name"] for entity in mr_entities if "Genre" in entity["labels"]
                           if entity["name"] in genres}
        mr_movie_genres = [(triple["head_uri"], genres.index(mr_uri_to_genre[triple["tail_uri"]]))
                           for triple in mr_triples
                           if triple["relation"] == "HAS_GENRE"
                           if triple["head_uri"] in joint_movies.values()
                           if triple["tail_uri"] in mr_uri_to_genre]
        dataset_movie_to_genres = {}
        for uri, genre in mr_movie_genres:
            if mr_to_idx[uri] not in dataset_movie_to_genres:
                dataset_movie_to_genres[mr_to_idx[uri]] = [genre]
            else:
                dataset_movie_to_genres[mr_to_idx[uri]].append(genre)

        # now, we need to associate each user to the genres she likes or dislikes
        # this is really similar to what we have done for movies

        mr_genre_ratings = [(rating["userId"], genres.index(mr_uri_to_genre[rating["uri"]]),
                             1 if rating["sentiment"] != -1 else 0)
                            for rating in mr_ratings_file
                            if rating["uri"] in mr_uri_to_genre
                            if rating["sentiment"] != 0]
        # this dict contains the genres that each user likes and the genres that each user dislikes
        dataset_user_to_genres = {}
        for user, genre, rating in mr_genre_ratings:
            if user not in user_mapping:  # some users have only rated movie genres
                user_mapping[user] = i
                i += 1
            if user_mapping[user] not in dataset_user_to_genres:
                dataset_user_to_genres[user_mapping[user]] = {"likes": [genre], "dislikes": []} \
                    if rating == 1 else {"likes": [], "dislikes": [genre]}
            else:
                if rating == 1:
                    dataset_user_to_genres[user_mapping[user]]["likes"].append(genre)
                else:
                    dataset_user_to_genres[user_mapping[user]]["dislikes"].append(genre)
