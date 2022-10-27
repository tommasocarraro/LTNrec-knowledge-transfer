import copy
import json
import pandas as pd
import os
import numpy as np
import sys
import string
import unidecode
from fuzzywuzzy import fuzz
from collections import Counter
from SPARQLWrapper import SPARQLWrapper, JSON, POST


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
        if not os.path.exists(os.path.join(self.data_path, "mindreader/processed/entities.csv")):
            self.fix_mr()
        if not os.path.exists(os.path.join(self.data_path, "ml-100k/processed/u.data")):
            self.fix_ml_100k()
        self.ml_100_movie_info = pd.read_csv(os.path.join(self.data_path, "ml-100k/processed/u.item"), sep="|",
                                             encoding="iso-8859-1", header=None).to_dict("records")
        self.ml_100_ratings = pd.read_csv(os.path.join(self.data_path, "ml-100k/processed/u.data"),
                                          sep="\t", header=None).to_dict("records")
        self.mr_entities = pd.read_csv(os.path.join(self.data_path,
                                                    "mindreader/processed/entities.csv")).to_dict("records")
        self.mr_triples = pd.read_csv(os.path.join(self.data_path, "mindreader/triples.csv")).to_dict("records")
        self.mr_ratings = pd.read_csv(os.path.join(self.data_path, "mindreader/ratings.csv")).to_dict("records")
        self.idx_to_uri = self.create_mapping()
        self.genres = ("Action Film", "Adventure Film", "Animated Film", "Children's Film", "Comedy Film", "Crime Film",
                       "Documentary Film", "Drama Film", "Fantasy Film", "Film Noir", "Horror Film", "Musical Film",
                       "Mystery Film", "Romance Film", "Science Fiction Film", "Thriller Film", "War Film",
                       "Western Film")
        self.create_intersection()
        self.create_union()

    def check(self):
        """
        Check the presence of all needed files:
        1. entities: contains the entities (movies, genres, actors, ...) of the MindReader KG;
        2. ratings: contains the user-entity ratings of the MindReader dataset;
        3. triples: contains the relationships between the entities of the MindReader KG. For example, who is the actor
        of the movie? Which is the genre? Which is the year? All this information is included in the triples file;
        4. u.data (MovieLens-100k): contains the user-item ratings of the MovieLens-100k dataset;
        5. u.item (MovieLens-100k): contains the content information of the movies in MovieLens-100k;
        6. movies (MovieLens-latest): contains the content information of the movies in the MovieLens latest dataset;
        7. links (MovieLens-latest): contains the links to iMDB for the movies in the dataset.
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

    def fix_ml_100k(self):
        """
        It removes duplicates in the ml-100k dataset. Some movies appear with different indexes because the same user
        has rated multiple times the same movie.
        It also recreates the indexing after these modifications have been done.
        It creates new csv files without the duplicates, both u.data and u.item, with the new indexing.
        """
        # get MovieLens-100k movies
        ml_100_movies_file = pd.read_csv(os.path.join(self.data_path, "ml-100k/u.item"), sep="|",
                                         encoding="iso-8859-1", header=None)
        ml_100_movies_file_dict = ml_100_movies_file.to_dict("records")

        # for each title, we get the list of corresponding indexes
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
            # create one unique index for the same movie with two different indexes
            # we keep the first of the two indexes
            ml_ratings[1] = ml_ratings[1].replace(idx[1], idx[0])
            # delete record with second occurrence of the movie in the file containing movie information
            ml_100_movies_file = ml_100_movies_file[ml_100_movies_file[0] != idx[1]]
            # aggregate ratings of users (we need to do that because now we have a user giving multiple
            # ratings to the same index)
            groups = ml_ratings.groupby(by=[0, 1])
            for i, (_, group) in enumerate(groups):
                if len(group) > 1:
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
                rating[0] = u
                u += 1
            else:
                rating[0] = user_mapping[rating[0]]
            if rating[1] not in item_mapping:
                item_mapping[rating[1]] = i
                rating[1] = i
                i += 1
            else:
                rating[1] = item_mapping[rating[1]]

        ml_ratings = pd.DataFrame.from_records(ml_ratings_dict)
        # correct indexes also on the file containing the movie info
        new_indexes = []
        for idx in ml_100_movies_file[0]:
            new_indexes.append(item_mapping[idx])
        ml_100_movies_file[0] = new_indexes
        ml_100_movies_file = ml_100_movies_file.sort_values(by=[0])

        # create new files
        ml_ratings.to_csv(os.path.join(self.data_path, "ml-100k/processed/u.data"), sep="\t", index=False, header=None)
        ml_100_movies_file.to_csv(os.path.join(self.data_path, "ml-100k/processed/u.item"), sep="|",
                                  encoding="iso-8859-1", index=False, header=None)

    def fix_mr(self):
        """
        It fixes the MindReader dataset by removing duplicated titles. There could be cases in which the title is the
        same even if the movie is slightly different, for example from a different year.

        To remove duplicates, we add the publication date to the title of the movie in brackets, as for the MovieLens
        datasets.
        """
        mr_entities = pd.read_csv(os.path.join(self.data_path, "mindreader/entities.csv"))
        mr_entities_dict = mr_entities.to_dict("records")
        mr_uri_to_title = {entity["uri"]: entity["name"] for entity in mr_entities_dict
                           if "Movie" in entity["labels"]}

        # get release dates of movies in MindReader
        query = "SELECT ?film ?publicationDate WHERE { ?film wdt:P577 ?publicationDate. VALUES ?film {" + \
                ' '.join(["wd:" + uri.split("/")[-1] for uri in mr_uri_to_title]) + \
                "} SERVICE wikibase:label { bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en'. } }"

        result = send_query(query)

        mr_uri_to_date = {}
        for match in result:
            uri = match['film']['value']
            date = match['publicationDate']['value'].split("-")[0]
            if uri not in mr_uri_to_date:
                # we take only the first date for each movie, some movies have multiple publication dates
                mr_uri_to_date[uri] = date

        # put date on the title when it is available
        mr_uri_to_title = {uri: "%s (%d)" % (title, int(mr_uri_to_date[uri]))
                           if uri in mr_uri_to_date else title for uri, title in mr_uri_to_title.items()}

        # recreate "name" column of original dataframe - the new column has now dates on the titles when they are
        # available
        mr_entities["name"] = [mr_uri_to_title[entity["uri"]]
                               if "Movie" in entity["labels"] else entity["name"] for entity in mr_entities_dict]

        mr_entities.to_csv(os.path.join(self.data_path, "mindreader/processed/entities.csv"), index=False)

    def create_mapping(self):
        """
        It creates the mapping between the indexes of MovieLens-100k and the URIs of MindReader.
        For doing that, it matches titles and years of the movies. In the cases in which there is not a perfect match
        (==), it tries to use fuzzy methods to check the similarity (~=) between the titles.

        The process is summarized as follows:
        1. the titles and years of MovieLens-100k are matched with titles and years of MovieLens-latest-small. This
        is done because the latter dataset has updated links to iMDB entities;
        2. the iMDB ids are given as input to a sparql query that retrieves the corresponding wikidata URIs;
        3. the match between MovieLens-100k idx and wikidata URIs contained in MindReader is created;
        4. it repeats the same procedure between titles and years of unmatched MovieLens-100k and MindReader movies.

        Returns a dictionary containing this mapping, namely ml_idx -> mr_uri.
        """
        if os.path.exists(os.path.join(self.data_path, "mapping.csv")):
            ml_100_idx_to_mr_uri = pd.read_csv("./datasets/mapping.csv").to_dict("records")
            ml_100_idx_to_mr_uri = {mapping["idx"]: mapping["uri"] for mapping in ml_100_idx_to_mr_uri}
        else:
            # create index to iMDB link dict
            ml_link_file = pd.read_csv(os.path.join(self.data_path, "ml-latest-small/links.csv"), dtype=str)
            ml_link_file["movieId"] = ml_link_file["movieId"].astype(np.int64)
            ml_link_file = ml_link_file.to_dict("records")
            ml_idx_to_link = {link["movieId"]: "tt" + link["imdbId"] for link in ml_link_file}

            # create title to idx dict
            # note that there are two "Emma" in MovieLens-latest, but only one is in MindReader. Other Emma has
            # been manually canceled
            ml_movies_file = pd.read_csv(os.path.join(self.data_path, "ml-latest-small/movies.csv")).to_dict("records")
            ml_title_to_idx = {movie["title"]: movie["movieId"] for movie in ml_movies_file}

            # get imdb id for movies in MovieLens-100k
            ml_100_title_to_idx = {movie[1]: movie[0] for movie in self.ml_100_movie_info}
            ml_100_idx_to_title = {idx: title for title, idx in ml_100_title_to_idx.items()}

            # match titles between MovieLens-100k and MovieLens-latest movies to get imdb id of MovieLens-100k movies
            ml_100_idx_to_imdb = {}
            matched = []
            for title, idx in ml_100_title_to_idx.items():
                if title in ml_title_to_idx:
                    # if title of 100k is in ml-latest dataset, get its link
                    ml_100_idx_to_imdb[idx] = ml_idx_to_link[ml_title_to_idx[title]]
                    matched.append(title)

            # for remaining unmatched movies, I use fuzzy methods since the titles have some marginal differences
            # (e.g., date, order of words)
            no_matched_100k = set(ml_100_title_to_idx.keys()) - set(matched)
            no_matched_ml = set(ml_title_to_idx.keys()) - set(matched)

            for title_100 in no_matched_100k:
                try:
                    # get year of ml-100k movie
                    f_year = int(title_100.strip()[-6:][1:5])
                except ValueError:
                    # if there is no year info, we do not find matches
                    continue
                best_sim = 0.0
                candidate_title_100 = ""
                candidate_title = ""
                for title in no_matched_ml:
                    try:
                        # get year of ml-latest movie
                        s_year = int(title.strip()[-6:][1:5])
                    except ValueError:
                        continue
                    # compute similarity between titles with year removed
                    title_sim = fuzz.token_set_ratio(title_100.strip()[:-6], title.strip()[:-6])
                    # search for best similarity between titles
                    if title_sim > best_sim and title_100 not in matched and title not in matched and f_year == s_year:
                        best_sim = title_sim
                        candidate_title_100 = title_100
                        candidate_title = title
                if best_sim >= 96:
                    # it seems that under 96 there are typos in the names and the matches are not reliable
                    # add movie to the dict containing the matches
                    ml_100_idx_to_imdb[ml_100_title_to_idx[candidate_title_100]] = \
                        ml_idx_to_link[ml_title_to_idx[candidate_title]]
                    if candidate_title_100 == candidate_title:
                        # match only one if they are the same
                        matched.append(candidate_title)
                    else:
                        # match both if they are different
                        matched.append(candidate_title_100)
                        matched.append(candidate_title)

            no_matched_100k = list(set(no_matched_100k) - set(matched))

            # now, we need to fetch the wikidata URIs for the movies we have matched between MovieLens-100k and
            # MovieLens-latest-small

            query = "SELECT ?label ?film WHERE {?film wdt:P345 ?label.VALUES ?label {" + \
                    ' '.join(["'" + imdb_url + "'" for idx, imdb_url in ml_100_idx_to_imdb.items()]) + \
                    "} SERVICE wikibase:label { bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en'. } }"

            result = send_query(query)

            imdb_to_wikidata = {match['label']['value']: match['film']['value'] for match in result}

            # create indexing from MovieLens-100k to Wikidata URIs - this indexing contains an entry only if the
            # previous query found the link between iMDB id and Wikidata entity - for 3 movies the link is not found
            # ml_100_idx_to_wikidata = {idx: imdb_to_wikidata[imdb_id] for idx, imdb_id in ml_100_idx_to_imdb.items()
            #                           if imdb_id in imdb_to_wikidata}
            ml_100_idx_to_wikidata = {}
            for idx, imdb_id in ml_100_idx_to_imdb.items():
                if imdb_id in imdb_to_wikidata:
                    ml_100_idx_to_wikidata[idx] = imdb_to_wikidata[imdb_id]
                else:
                    no_matched_100k.append(ml_100_idx_to_title[idx])
                    matched.remove(ml_100_idx_to_title[idx])

            # filter mapping based on presence of movie in MindReader
            mr_uri_to_title = {entity["uri"]: entity["name"] for entity in self.mr_entities
                               if "Movie" in entity["labels"]}
            mr_title_to_uri = {title: uri for uri, title in mr_uri_to_title.items()}
            # the match is kept only if the retrieved wikidata URI is in the corpus of movies of MindReader
            ml_100_idx_to_mr_uri = {}
            for idx, uri in ml_100_idx_to_wikidata.items():
                if uri in mr_uri_to_title:
                    ml_100_idx_to_mr_uri[idx] = uri
                else:
                    no_matched_100k.append(ml_100_idx_to_title[idx])
                    matched.remove(ml_100_idx_to_title[idx])

            no_matched_mr = set(mr_uri_to_title.keys()) - set(ml_100_idx_to_mr_uri.values())

            # repeat fuzzy procedure between MovieLens-100k and MindReader directly, since there could be some errors
            # in the links of the ml-latest dataset

            for title_100 in no_matched_100k:
                try:
                    # get year of ml-100k movie
                    f_year = int(title_100.strip()[-6:][1:5])
                except ValueError:
                    continue
                best_sim = 0.0
                candidate_title_100 = ""
                candidate_mr_title = ""
                for uri in no_matched_mr:
                    try:
                        # get year of MindReader movie
                        s_year = int(mr_uri_to_title[uri].strip()[-6:][1:5])
                    except ValueError:
                        continue
                    # compute similarity between titles with year removed
                    title_sim = fuzz.token_set_ratio(title_100.strip()[:-6], mr_uri_to_title[uri].strip()[:-6])
                    # search for best similarity between titles
                    if title_sim > best_sim and title_100 not in matched and mr_uri_to_title[uri] not in matched \
                            and f_year == s_year:
                        best_sim = title_sim
                        candidate_title_100 = title_100
                        candidate_mr_title = mr_uri_to_title[uri]
                if best_sim >= 96:
                    ml_100_idx_to_mr_uri[ml_100_title_to_idx[candidate_title_100]] = mr_title_to_uri[candidate_mr_title]
                    matched.append(candidate_title_100)
                    matched.append(candidate_mr_title)

            mapping = pd.DataFrame.from_dict({"idx": list(ml_100_idx_to_mr_uri.keys()),
                                              "uri": list(ml_100_idx_to_mr_uri.values())})
            mapping.to_csv("./datasets/mapping.csv", index=False)

        return ml_100_idx_to_mr_uri

    def create_intersection(self):
        """
        It creates the dataset that is the intersection between the movies of MovieLens-100k and the movies of
        MindReader. We have ratings from users of MovieLens and ratings from users of MindReader. For MindReader, we
        also have ratings on the genres of the movies.

        Only the most popular genres are taken into account, namely the genres appearing in the MovieLens-100k dataset.
        The ratings on the other genres are not considered since they are sub-genres of the main genres.

        The idea of this dataset is to test if the information about the genres helps in improving the accuracy on
        the MovieLens-100k. It is a kind of knowledge transfer from MindReader to MovieLens-100k.

        We need to pay attention to the fact that it could be the information provided by the movie ratings of
        MindReader to increase the performance. For this reason, an ablation study should be done. We need to train both
        with genres and without genres.
        """
        # create a new unified indexing. The same movie on MovieLens-100k and MindReader has different indexes
        # we need to associate them to the same index
        # create unique indexing
        ml_to_idx = {idx: i for i, idx in enumerate(self.idx_to_uri.keys())}
        mr_to_idx = {uri: i for i, uri in enumerate(self.idx_to_uri.values())}
        # create inverse mapping too, this could be helpful to go from one indexing to the other
        idx_to_ml = {idx: ml_idx for ml_idx, idx in ml_to_idx.items()}
        idx_to_mr = {idx: uri for uri, idx in mr_to_idx.items()}

        # take only ratings of the 100k movies that are in the joint dataset
        # create implicit ratings (pos if rating >= 4, 0 otherwise)
        ml_ratings = np.array([(rating[0], rating[1], 1 if rating[2] >= 4 else 0) for rating in self.ml_100_movie_info
                               if rating[1] in self.idx_to_uri])

        # take MindReader ratings only for movies in the joint dataset
        # remove the unknown ratings (ratings equal to 0)
        # map pos ratings to 1 and neg rating to 0
        mr_ratings = [(rating["userId"], rating["uri"], rating["sentiment"] if rating["sentiment"] != -1 else 0)
                      for rating in self.mr_ratings
                      if rating["uri"] in self.idx_to_uri.values()
                      if rating["sentiment"] != 0]
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
        # in this case, we can use the genres of MindReader since this is the intersection dataset
        # in the union, we need to use also the genres of MovieLens-100k, since we have some movies for which we do not
        # have the information in MindReader
        # we use only the most common genres, to reduce sparsity (genres of MovieLens-100k)
        mr_entities = pd.read_csv(os.path.join(self.data_path, "mindreader/processed/entities.csv")).to_dict("records")
        mr_triples = pd.read_csv(os.path.join(self.data_path, "mindreader/triples.csv")).to_dict("records")
        # take the URIs of selected genres
        mr_uri_to_genre = {entity["uri"]: entity["name"] for entity in mr_entities if "Genre" in entity["labels"]
                           if entity["name"] in self.genres}
        # dict which associates each movie to its genres (index of the genre inside genres, this is used
        # for one-hot encoding)
        mr_movie_genres = [(triple["head_uri"], self.genres.index(mr_uri_to_genre[triple["tail_uri"]]))
                           for triple in mr_triples
                           if triple["relation"] == "HAS_GENRE"
                           if triple["head_uri"] in self.idx_to_uri.values()
                           if triple["tail_uri"] in mr_uri_to_genre]
        dataset_movie_to_genres = {}
        for uri, genre in mr_movie_genres:
            if mr_to_idx[uri] not in dataset_movie_to_genres:
                dataset_movie_to_genres[mr_to_idx[uri]] = [genre]
            else:
                dataset_movie_to_genres[mr_to_idx[uri]].append(genre)

        # now, we need to associate each user to the genres she likes or dislikes
        # this is really similar to what we have done for movies
        # get ratings of users of MindReader for the genres
        mr_genre_ratings = [(rating["userId"], self.genres.index(mr_uri_to_genre[rating["uri"]]),
                             1 if rating["sentiment"] != -1 else 0)
                            for rating in mr_ratings_file
                            if rating["uri"] in mr_uri_to_genre
                            if rating["sentiment"] != 0]
        # this dict contains the genres that each user likes and the genres that each user dislikes
        dataset_user_to_genres = {}
        for user, genre, rating in mr_genre_ratings:
            if user not in user_mapping:  # some users have only rated movie genres, so we need to add them to the map
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

        return dataset_ratings, dataset_movie_to_genres, dataset_user_to_genres

    def create_union(self):
        """
        It creates the dataset that is the union between the movies of MovieLens-100k and the movies of
        MindReader. We have ratings from users of MovieLens and ratings from users of MindReader. For MindReader, we
        also have ratings on the genres of the movies.

        Only the most popular genres are taken into account, namely the genres appearing in the MovieLens-100k dataset.
        The ratings on the other genres are not considered since they are sub-genres of the main genres.

        The idea of this dataset is to test if the information about the genres helps in improving the accuracy on
        the MovieLens-100k, especially in the movies that do not belong to the intersection of the two datasets.
        It is a kind of knowledge transfer from MindReader to MovieLens-100k.

        We need to pay attention to the fact that it could be the information provided by the movie ratings of
        MindReader to increase the performance. For this reason, an ablation study should be done. We need to train both
        with genres and without genres.
        """
        # for creating the union, I need to take all the movies of MovieLens and all the movies of MindReader
        # I still have to create a unique indexing. Some movies will be map to the same idx, some not
        # idea: I create indexes for joint movies between the datasets, then I create indexes for the remaining movies
        # in MovieLens-100k and MindReader
        # create unique indexing for shared movies between datasets
        ml_to_idx = {idx: i for i, idx in enumerate(self.idx_to_uri.keys())}
        mr_to_idx = {uri: i for i, uri in enumerate(self.idx_to_uri.values())}
        # create indexing for movies which are not shared
        # take movies of MovieLens-100k
        ml_movies = pd.read_csv(os.path.join(self.data_path, "ml-100k/processed/u.item"), sep="|",
                                encoding="iso-8859-1", header=None).to_dict("records")
        # take MovieLens-100k ratings
        ml_ratings_file = pd.read_csv(os.path.join(self.data_path, "ml-100k/u.data"),
                                      sep="\t", header=None).to_dict("records")
        # create implicit ratings (pos if rating >= 4, 0 otherwise)
        ml_ratings = np.array([(rating[0], rating[1], 1 if rating[2] >= 4 else 0) for rating in ml_ratings_file])
        set_rated_ml = set([i for _, i, _ in ml_ratings])
        # create indexes for ml-100k movie not in the joint set
        i = len(self.idx_to_uri)
        for movie in ml_movies:
            if movie[0] not in self.idx_to_uri and movie[0] in set_rated_ml:
                ml_to_idx[movie[0]] = i
                i += 1

        # take MindReader ratings
        mr_ratings_file = pd.read_csv(os.path.join(self.data_path, "mindreader/ratings.csv")).to_dict("records")
        # take MindReader movie URIs
        mr_entities = pd.read_csv(os.path.join(self.data_path, "mindreader/processed/entities.csv")).to_dict("records")
        mr_movies = [entity["uri"] for entity in mr_entities if "Movie" in entity["labels"]]
        # remove the unknown ratings (ratings equal to 0)
        # map pos ratings to 1 and neg rating to 0
        mr_ratings = [(rating["userId"], rating["uri"], rating["sentiment"] if rating["sentiment"] != -1 else 0)
                      for rating in mr_ratings_file
                      if rating["sentiment"] != 0
                      if rating["uri"] in mr_movies]
        set_rated_mr = set([i for _, i, _ in mr_ratings])
        for movie in mr_movies:
            if movie not in self.idx_to_uri.values() and movie in set_rated_mr:
                mr_to_idx[movie] = i
                i += 1

        # create inverse mapping too, this could be helpful to go from one dataset to the other
        idx_to_ml = {idx: ml_idx for ml_idx, idx in ml_to_idx.items()}
        idx_to_mr = {idx: uri for uri, idx in mr_to_idx.items()}

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
        # in the union we need to use also the genres of MovieLens-100k, since we have some movies for which we do not
        # have the information in MindReader
        # we use only the most common genres, to reduce sparsity (genres of MovieLens-100k)

        genres = ("Action Film", "Adventure Film", "Animated Film", "Children's Film", "Comedy Film", "Crime Film",
                  "Documentary Film", "Drama Film", "Fantasy Film", "Film Noir", "Horror Film", "Musical Film",
                  "Mystery Film", "Romance Film", "Science Fiction Film", "Thriller Film", "War Film", "Western Film")
        mr_triples = pd.read_csv(os.path.join(self.data_path, "mindreader/triples.csv")).to_dict("records")
        # take the URIs of selected genres
        mr_uri_to_genre = {entity["uri"]: entity["name"] for entity in mr_entities if "Genre" in entity["labels"]
                           if entity["name"] in genres}
        # dict which associate each movie to its genres (index of the genre inside genres, this is used for one-hot enc)
        mr_movie_genres = [(triple["head_uri"], genres.index(mr_uri_to_genre[triple["tail_uri"]]))
                           for triple in mr_triples
                           if triple["relation"] == "HAS_GENRE"
                           if triple["tail_uri"] in mr_uri_to_genre]
        # get genres of movies in MovieLens-100k that are not in the joint set
        ml_movie_genres = []
        with open(os.path.join(self.data_path, "ml-100k/u.item"), encoding="iso-8859-1") as fp:
            for line in fp:
                idx = int(line.split("|")[0])
                if idx not in self.idx_to_uri:
                    ml_movie_genres.extend([(idx, genre_idx)
                                            for genre_idx, genre in enumerate(line[-36:].split("|"))
                                            if int(genre) == 1])

        dataset_movie_to_genres = {}
        for uri, genre in mr_movie_genres:
            if mr_to_idx[uri] not in dataset_movie_to_genres:
                dataset_movie_to_genres[mr_to_idx[uri]] = [genre]
            else:
                dataset_movie_to_genres[mr_to_idx[uri]].append(genre)

        for idx, genre in ml_movie_genres:
            if ml_to_idx[idx] not in dataset_movie_to_genres:
                dataset_movie_to_genres[ml_to_idx[idx]] = [genre]
            else:
                dataset_movie_to_genres[ml_to_idx[idx]].append(genre)

        # now, we need to associate each user to the genres she likes or dislikes - this info is only on MindReader
        # get ratings of users of MindReader for the genres
        mr_genre_ratings = [(rating["userId"], genres.index(mr_uri_to_genre[rating["uri"]]),
                             1 if rating["sentiment"] != -1 else 0)
                            for rating in mr_ratings_file
                            if rating["uri"] in mr_uri_to_genre
                            if rating["sentiment"] != 0]
        # this dict contains the genres that each user likes and the genres that each user dislikes
        dataset_user_to_genres = {}
        for user, genre, rating in mr_genre_ratings:
            if user not in user_mapping:  # some users have only rated movie genres, so we need to add them to the map
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

        return dataset_ratings, dataset_movie_to_genres, dataset_user_to_genres
