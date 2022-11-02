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
import random
from SPARQLWrapper import SPARQLWrapper, JSON, POST


# todo ci sono 457 film matchati tra i due dataset per i quali pero' non ci sono ratings su mindreader, quindi e' un match inutile perche' non abbiamo un test set e nemmeno una ground truth per quei film
# todo bisognerebbe discutere se possano essere comunque utili con la scusa dei generi, magari quelli propogano informazione, pero' non avendo quei film dei rating, l'informazione si propaga solo attraverso il LikesGenre appreso
# todo discutere di questa cosa con Luciano e Alessandro
# todo moltissimi film su mindreader non hanno ratings o hanno solo ratings unknown, cosa posso fare? -> possono essere usati come cold start in casi futuri, pero' non ho rating su quelli e quindi non posso fare dei test set -> per ora li ho eliminati tutti
# todo il problema e' che molti di questi facevano parte dell'intersezione tra i due dataset
# todo osservazione interessante: alcuni film non hanno ratings, potrebbero essere trattati come dei casi di cold-start in futuro, e' che non li puoi valutare
# todo per simulare il cold-start si possono prendere tutti i rating di un film e metterli in test, in modo tale che durante il training non ci siano piu' rating per quel film


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
        self.genres = ("Action Film", "Adventure Film", "Animated Film", "Children's Film", "Comedy Film", "Crime Film",
                       "Documentary Film", "Drama Film", "Fantasy Film", "Film Noir", "Horror Film", "Musical Film",
                       "Mystery Film", "Romance Film", "Science Fiction Film", "Thriller Film", "War Film",
                       "Western Film")
        if not os.path.exists(os.path.join(self.data_path, "ml-100k/processed/item_mapping.csv")):
            self.process_ml_100k()
        if not os.path.exists(os.path.join(self.data_path, "mindreader/processed/item_mapping.csv")):
            self.process_mr()
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
        self.ml_to_mr = self.create_mapping()

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

    def process_ml_100k(self, threshold=4):
        """
        It removes duplicates in the ml-100k dataset. Some movies appear with different indexes because the same user
        has rated multiple times the same movie.
        It also recreates the indexing after these modifications have been done.
        It creates new csv files without the duplicates, both u.data and u.item, with the new indexing.

        :param threshold: a threshold used to create implicit feedbacks from the explicit feedbacks of ml-100k
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
                u += 1
            if rating[1] not in item_mapping:
                item_mapping[rating[1]] = i
                i += 1
            rating[0] = user_mapping[rating[0]]
            rating[1] = item_mapping[rating[1]]
            rating[2] = 1 if rating[2] >= threshold else 0

        ml_ratings = pd.DataFrame.from_records(ml_ratings_dict)
        # correct indexes also on the file containing the movie info
        new_indexes = []
        for _, movie_idx in ml_100_movies_file[0].items():
            new_indexes.append(item_mapping[movie_idx])
        ml_100_movies_file[0] = new_indexes
        ml_100_movies_file = ml_100_movies_file.sort_values(by=[0])
        ml_100_movies_file.reset_index()

        # process genres and get indexes
        ml_100_movies_records = []
        for row_idx, row in ml_100_movies_file.iterrows():
            movie_genres = [genre for _, genre in row[6:].items()]
            movie_genres_idx = [str(idx) for idx, genre in enumerate(movie_genres) if genre == 1]
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
        It fixes the MindReader dataset by removing duplicated titles. There could be cases in which the title is the
        same even if the movie is slightly different, for example from a different year.
        To remove duplicates, we add the publication date to the title of the movie in brackets, as for the MovieLens
        datasets.
        After these modifications have been done, it recreates the indexing of MindReader by substituting URIs with
        proper indexes. The same is done for the user identifiers.
        """
        mr_entities = pd.read_csv(os.path.join(self.data_path, "mindreader/entities.csv"))
        mr_entities_dict = mr_entities.to_dict("records")
        # get titles of movies
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

        # create new indexing for the dataset (users, movies)
        mr_ratings = pd.read_csv(os.path.join(self.data_path, "mindreader/ratings.csv"))
        mr_entities_dict = mr_entities.to_dict("records")
        mr_movies = [entity["uri"] for entity in mr_entities_dict if "Movie" in entity["labels"]]
        # we want ratings only for the allowed genres
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
            rating["sentiment"] = int(rating["sentiment"] > 0)
            mr_movie_ratings_new.append(
                {"u_idx": rating["userId"], "i_idx": rating["uri"], "rate": rating["sentiment"]})
        mr_movie_ratings_new = pd.DataFrame.from_records(mr_movie_ratings_new)
        mr_movie_ratings_new.to_csv(os.path.join(self.data_path, "mindreader/processed/movie_ratings.csv"), index=False)

        mr_genre_ratings_new = []
        for rating in mr_genre_ratings:
            if rating["userId"] not in user_mapping:
                user_mapping[rating["userId"]] = u
                u += 1
            rating["userId"] = user_mapping[rating["userId"]]
            rating["uri"] = self.genres.index(mr_genres[rating["uri"]])
            rating["sentiment"] = int(rating["sentiment"] > 0)
            mr_genre_ratings_new.append(
                {"u_idx": rating["userId"], "g_idx": rating["uri"], "rate": rating["sentiment"]})
        mr_genre_ratings_new = pd.DataFrame.from_records(mr_genre_ratings_new)
        mr_genre_ratings_new.to_csv(os.path.join(self.data_path, "mindreader/processed/genre_ratings.csv"), index=False)

        # now, we need to change the URIs in the triple.csv file and create e new file
        mr_triples_dict = pd.read_csv(os.path.join(self.data_path, "mindreader/triples.csv")).to_dict("records")
        mr_movie_genre_dict = {}
        for triple in mr_triples_dict:
            if triple["head_uri"] in item_mapping and triple["tail_uri"] in mr_genres \
                    and triple["relation"] == "HAS_GENRE":
                if item_mapping[triple["head_uri"]] not in mr_movie_genre_dict:
                    mr_movie_genre_dict[item_mapping[
                        triple["head_uri"]]] = [self.genres.index(mr_genres[triple["tail_uri"]])]
                else:
                    mr_movie_genre_dict[item_mapping[
                        triple["head_uri"]]].append(self.genres.index(mr_genres[triple["tail_uri"]]))

        # now, we need to change the URIs in the entities.csv file and create a new file
        # the new file contains also the genres of the movies, as in MovieLens
        # if a movie is not in mr_movie_genre_dict, it means that the movie has not ratings, so we do not include it
        # in the entities since it is not necessary for the purpose of our experiments
        # there could be a movie that is rated but does not belong to any of the selected genres
        mr_entities_new = []
        for entity in mr_entities_dict:
            if entity["uri"] in item_mapping:
                mr_entities_new.append({"idx": item_mapping[entity["uri"]],
                                        "title": entity["name"],
                                        "genres": "|".join([str(g)
                                                            for g in mr_movie_genre_dict[item_mapping[entity["uri"]]]
                                                            ])
                                        if item_mapping[entity["uri"]] in mr_movie_genre_dict else "None"})
        mr_entities_new = pd.DataFrame.from_records(mr_entities_new)
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
            ml_100_idx_to_mr_idx = pd.read_csv("./datasets/mapping.csv").to_dict("records")
            ml_100_idx_to_mr_idx = {mapping["ml_idx"]: mapping["mr_idx"] for mapping in ml_100_idx_to_mr_idx}
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
            mr_uri_to_idx = pd.read_csv(os.path.join(self.data_path,
                                                     "mindreader/processed/item_mapping.csv")).to_dict("records")
            mr_uri_to_idx = {mapping["old_idx"]: mapping["new_idx"] for mapping in mr_uri_to_idx}
            mr_idx_to_uri = {idx: uri for uri, idx in mr_uri_to_idx.items()}
            mr_uri_to_title = {mr_idx_to_uri[movie["idx"]]: movie["title"] for movie in self.mr_movie_info}
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
                    if candidate_title_100 == candidate_mr_title:
                        matched.append(candidate_title_100)
                    else:
                        matched.append(candidate_title_100)
                        matched.append(candidate_mr_title)

            ml_100_idx_to_mr_idx = {idx: mr_uri_to_idx[uri] for idx, uri in ml_100_idx_to_mr_uri.items()}

            mapping = pd.DataFrame.from_dict({"ml_idx": list(ml_100_idx_to_mr_idx.keys()),
                                              "mr_idx": list(ml_100_idx_to_mr_idx.values())})
            mapping.to_csv("./datasets/mapping.csv", index=False)

        return ml_100_idx_to_mr_idx

    def create_ml_mr_fusion(self):
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
        ml_to_new_idx = {ml_idx: idx for idx, ml_idx in enumerate(self.ml_to_mr.keys())}
        mr_to_new_idx = {mr_idx: idx for idx, mr_idx in enumerate(self.ml_to_mr.values())}
        # create indexing for movies which are not shared
        ml_ratings = np.array([(rating[0], rating[1], rating[2]) for rating in self.ml_100_ratings])
        # create indexes for ml-100k movie not in the joint set
        i = len(self.ml_to_mr)
        for movie in self.ml_100_movie_info:
            if movie[0] not in self.ml_to_mr:
                ml_to_new_idx[movie[0]] = i
                i += 1
        # create indexes for ml-100k movie not in the joint set
        mr_movie_ratings = np.array([(rating["u_idx"], rating["i_idx"], rating["rate"])
                                     for rating in self.mr_movie_ratings])
        for movie in self.mr_movie_info:
            if movie["idx"] not in self.ml_to_mr.values():
                mr_to_new_idx[movie["idx"]] = i
                i += 1

        # create unique user indexing
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
        dataset_ratings = [(user_mapping_ml[u], ml_to_new_idx[i], r) for u, i, r in ml_ratings]
        dataset_ratings.extend([(user_mapping_mr[u], mr_to_new_idx[i], r) for u, i, r in mr_movie_ratings])
        dataset_ratings = np.array(dataset_ratings)

        # we need to associate each movie to its genres
        # in the union we need to use also the genres of MovieLens-100k, since we have some movies for which we do not
        # have the information in MindReader
        # we use only the most common genres, to reduce sparsity (genres of MovieLens-100k)
        # for the joint set of movies, we use the genres in MindReader
        dataset_movie_to_genres = {mr_to_new_idx[movie["idx"]]: movie["genres"].split("|")
                                   for movie in self.mr_movie_info}
        dataset_movie_to_genres = dataset_movie_to_genres | {ml_to_new_idx[movie[0]]: movie[2].split("|")
                                                             for movie in self.ml_100_movie_info
                                                             if ml_to_new_idx[movie[0]] not in dataset_movie_to_genres}

        # now, we need to associate each user to the genres she likes or dislikes - this info is only on MindReader
        # get ratings of users of MindReader for the genres
        mr_genre_ratings = np.array([(rating["u_idx"], rating["g_idx"], rating["rate"])
                                     for rating in self.mr_genre_ratings])
        # this dict contains the genres that each user likes and the genres that each user dislikes
        dataset_user_to_genres = {}
        for u, g, r in mr_genre_ratings:
            if u not in user_mapping_mr:  # some users have only rated movie genres, so we need to add them to the map
                user_mapping_mr[u] = i
                i += 1
            if user_mapping_mr[u] not in dataset_user_to_genres:
                dataset_user_to_genres[user_mapping_mr[u]] = {"likes": [g], "dislikes": []} \
                    if r == 1 else {"likes": [], "dislikes": [g]}
            else:
                if r == 1:
                    dataset_user_to_genres[user_mapping_mr[u]]["likes"].append(g)
                else:
                    dataset_user_to_genres[user_mapping_mr[u]]["dislikes"].append(g)

        return dataset_ratings, dataset_movie_to_genres, dataset_user_to_genres, ml_to_new_idx, mr_to_new_idx, \
               user_mapping_ml, user_mapping_mr

    def get_ml100k_folds(self, seed):
        """
        Creates and returns the training, validation, and test set of the MovieLens-100k dataset.

        The procedure starts from the entire datasets. To create the test set, one random positive movie rating for
        each user is held-out. To complete the test example of the user, 100 irrelevant movies are randomly sampled.
        The metrics evaluate the recommendation based on the position of the positive movie in a ranking containing all
        the 101 movies.

        The validation set is constructed in the same way, starting from the remaining rating in the dataset.

        The seed parameter is used for reproducibility of experiments.
        """
        # set seed
        random.seed(seed)
        ratings = pd.DataFrame.from_records(self.ml_100_ratings)
        # group by user idx
        groups = ratings.groupby(by=[0])
        # get set of movie indexes
        movies = set(ratings[1].unique())
        validation = []
        test = []
        for user_idx, group in groups:
            # positive ratings for the user
            group_pos = group[group[2] == 1]
            # check if it is possible to sample
            if len(group_pos):
                # take one random positive rating for test and one for validation
                sampled_pos = group_pos.sample(n=2, random_state=seed)
                # remove sampled ratings from the dataset
                ratings.drop(sampled_pos.index, inplace=True)
                sampled_pos = list(sampled_pos[1])
                # remove sampled ratings from the group
                seen_without_test = group_pos[~group_pos[1].isin(sampled_pos)]
                # take 100 randomly sampled negative (not seen by the user) movies for test and validation
                not_seen = movies - set(seen_without_test[1])
                neg_movies_test = random.sample(not_seen, 100)
                neg_movies_val = random.sample(not_seen, 100)
                validation.append([[user_idx, item] for item in neg_movies_val + [sampled_pos[0]]])
                test.append([[user_idx, item] for item in neg_movies_test + [sampled_pos[1]]])

        validation = np.array(validation)
        test = np.array(test)

        # create np array of training ratings
        train = ratings.to_dict("records")
        train = np.array([(rating[0], rating[1], rating[2]) for rating in train])

        return train, validation, test
