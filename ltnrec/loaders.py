import random
import numpy as np
import torch
import ltn
import pandas as pd
import itertools


class TrainingDataLoaderLTNRegression:
    """
    Data loader to load the training set of the dataset. It creates batches and wrap them inside LTN
    variables ready for the learning.
    """

    def __init__(self,
                 data,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the training data loader.
        :param data: list of triples (user, item, rating)
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        self.data = np.array(data)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.data[idxlist[start_idx:end_idx]]

            yield ltn.Variable('users', torch.tensor(data[:, 0]), add_batch_dim=False), \
                  ltn.Variable('items', torch.tensor(data[:, 1]), add_batch_dim=False), \
                  ltn.Variable('ratings', torch.tensor(data[:, -1]), add_batch_dim=False)


class TrainingDataLoaderLTNClassification:
    """
    Data loader to load the training set of the dataset. It creates batches and wrap them inside LTN
    variables ready for the learning.
    """

    def __init__(self,
                 data,
                 non_relevant_sampling=False,
                 n_users=None,
                 n_items=None,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the training data loader.
        :param data: list of triples (user, item, rating)
        :param non_relevant_sampling: whether the batch has to include non-relevant user-item interactions. This is
        useful when the loader is used to train a model that uses transfer learning. The transfer learning is applied
        to these non-relevant interactions (holes of the user-item matrix)
        :param n_users: number of users in the dataset. It is used when non_relevent_sampling is True
        :param n_items: number of items in the dataset. It is used when non_relevant_sampling is True
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        if non_relevant_sampling:
            assert n_users is not None and n_items is not None, "When non_relevant_sampling is True, n_users and " \
                                                                "n_items are required to perform the sampling."
        self.data = np.array(data)
        self.non_relevant_sampling = non_relevant_sampling
        self.n_users = n_users
        self.n_items = n_items
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.data[idxlist[start_idx:end_idx]]
            pos_ex = data[data[:, -1] == 1]
            neg_ex = data[data[:, -1] != 1]
            if self.non_relevant_sampling:
                # get some non-relevant user-item pairs - the probability of sampling real user-item pairs is very low
                # due to the sparsity of a recommendation dataset
                u_idx = np.random.choice(self.n_users, data.shape[0])
                i_idx = np.random.choice(self.n_items, data.shape[0])

            if len(pos_ex) and len(neg_ex):
                if not self.non_relevant_sampling:
                    yield ltn.Variable('users_pos', torch.tensor(pos_ex[:, 0]), add_batch_dim=False), \
                          ltn.Variable('items_pos', torch.tensor(pos_ex[:, 1]), add_batch_dim=False),\
                          ltn.Variable('users_neg', torch.tensor(neg_ex[:, 0]), add_batch_dim=False),\
                          ltn.Variable('items_neg', torch.tensor(neg_ex[:, 1]), add_batch_dim=False)
                else:
                    yield ltn.Variable('users_pos', torch.tensor(pos_ex[:, 0]), add_batch_dim=False), \
                          ltn.Variable('items_pos', torch.tensor(pos_ex[:, 1]), add_batch_dim=False), \
                          ltn.Variable('users_neg', torch.tensor(neg_ex[:, 0]), add_batch_dim=False), \
                          ltn.Variable('items_neg', torch.tensor(neg_ex[:, 1]), add_batch_dim=False), \
                          ltn.Variable('users', torch.tensor(u_idx), add_batch_dim=False), \
                          ltn.Variable('items', torch.tensor(i_idx), add_batch_dim=False)
            elif len(pos_ex):
                if not self.non_relevant_sampling:
                    yield ltn.Variable('users_pos', torch.tensor(pos_ex[:, 0]), add_batch_dim=False), \
                          ltn.Variable('items_pos', torch.tensor(pos_ex[:, 1]), add_batch_dim=False), None, None
                else:
                    yield ltn.Variable('users_pos', torch.tensor(pos_ex[:, 0]), add_batch_dim=False), \
                          ltn.Variable('items_pos', torch.tensor(pos_ex[:, 1]), add_batch_dim=False), None, None, \
                          ltn.Variable('users', torch.tensor(u_idx), add_batch_dim=False), \
                          ltn.Variable('items', torch.tensor(i_idx), add_batch_dim=False)
            else:
                if not self.non_relevant_sampling:
                    yield None, None, ltn.Variable('users_neg', torch.tensor(neg_ex[:, 0]), add_batch_dim=False),\
                          ltn.Variable('items_neg', torch.tensor(neg_ex[:, 1]), add_batch_dim=False)
                else:
                    yield None, None, ltn.Variable('users_neg', torch.tensor(neg_ex[:, 0]), add_batch_dim=False), \
                          ltn.Variable('items_neg', torch.tensor(neg_ex[:, 1]), add_batch_dim=False), \
                          ltn.Variable('users', torch.tensor(u_idx), add_batch_dim=False), \
                          ltn.Variable('items', torch.tensor(i_idx), add_batch_dim=False)


class TrainingDataLoaderLTNClassificationSampling:
    """
    Data loader to load the training set of the dataset. It creates batches and wrap them inside LTN
    variables ready for the learning.
    """

    def __init__(self,
                 data,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the training data loader.
        :param data: list of triples (user, item, rating)
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        data = np.array(data)
        self.pos_data = data[data[:, -1] == 1]
        self.neg_data = data[data[:, -1] != 1]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.pos_data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.pos_data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            pos_data = self.pos_data[idxlist[start_idx:end_idx]]
            # sample some negative data in such a way positive and negative examples are balanced
            neg_idx = random.sample(range(len(self.neg_data)), len(pos_data))
            neg_data = self.neg_data[neg_idx]

            yield ltn.Variable('users_pos', torch.tensor(pos_data[:, 0]), add_batch_dim=False), \
                  ltn.Variable('items_pos', torch.tensor(pos_data[:, 1]), add_batch_dim=False), \
                  ltn.Variable('users_neg', torch.tensor(neg_data[:, 0]), add_batch_dim=False), \
                  ltn.Variable('items_neg', torch.tensor(neg_data[:, 1]), add_batch_dim=False)


class TrainingDataLoaderLTNBPR:
    """
    Data loader designed to provide batches for learning a MF model with Bayesian Personalized Ranking. It is custom in
    the sense that it is designed to work with explicit feedback. The prepare_data method prepares the dataset for
    learning a model with BPR using both positive and negative interactions. For each user, for each positive
    interaction, this loaders matches the positive interaction with all the negative interactions of the user.
    """

    def __init__(self,
                 data,
                 u_i_matrix,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the training data loader.
        :param data: list of triples (user, item, rating)
        :param u_i_matrix: sparse user-item interaction matrix with 1 if user has interacted the item and 0 otherwise.
        It is used internally to sample non-relevant items for the training user that do not have negative items.
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        self.u_i_matrix = u_i_matrix
        self.data = self.prepare_data(data)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def prepare_data(self, data):
        data = pd.DataFrame(data, columns=['userId', 'uri', 'sentiment'])
        training_triples = []
        users = data.groupby(["userId"])
        for user, user_ratings in users:
            pos_ratings = user_ratings[user_ratings["sentiment"] == 1]
            neg_ratings = user_ratings[user_ratings["sentiment"] != 1]
            # check that at least one example can be constructed for the current user
            if len(pos_ratings) >= 1 and len(neg_ratings) >= 1:
                training_triples.extend([[user, pos, neg] for pos, neg in list(itertools.product(pos_ratings["uri"],
                                                                                                 neg_ratings["uri"]))])
            else:
                # here, we sample one non-relevant item for each positive interaction of the user since we do not have
                # negative items for the current user
                # the probability of sampling a negative item is really high due to the high sparsity
                # very often, we can assume that non-relevant items are negative items
                # the problem is that they remain the same for all epochs, so there could be a bias on the selected
                # items. Performing experiments with more seeds is imperative in this case
                # todo there is a user for which we do not have examples because he has only negative examples in the
                #  training set. The validation on this user will be very challening but it is just a user
                if len(pos_ratings) >= 1 and len(neg_ratings) == 0:
                    training_triples.extend([[user, pos, random.choice(list(set(range(self.u_i_matrix.shape[1])) -
                                             set(self.u_i_matrix[user].nonzero()[1])))]
                                             for pos in list(pos_ratings["uri"])])
        return np.array(training_triples)

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            triples = self.data[idxlist[start_idx:end_idx]]
            yield ltn.Variable('users', torch.tensor(triples[:, 0]), add_batch_dim=False), \
                  ltn.Variable('items_pos', torch.tensor(triples[:, 1]), add_batch_dim=False), \
                  ltn.Variable('items_neg', torch.tensor(triples[:, 2]), add_batch_dim=False)


class TrainingDataLoaderLTNGenres:
    """
    Data loader to load the training set of the dataset. It creates batches and wrap them inside LTN
    variables ready for the learning. This loader differs from the TrainingDataLoaderLTN. In particular, it creates
    LTN variables to reason on formulas which involve the genres of the movies.
    """
    def __init__(self,
                 movie_ratings,
                 n_users,
                 n_items,
                 n_genres,
                 genre_sample_size=5,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the training data loader.
        :param movie_ratings: list of triples (user, item, rating)
        :param n_users: number of users in the dataset
        :param n_items: number of items in the dataset - this cannot be computed from training ratings since the
        procedure which creates the folds could have moved the only rating of one item in test set
        :param n_genres: number of movie genres in the dataset
        :param genre_sample_size: number of movie genres that have to be sampled at each batch (we sample to increase
        efficiency)
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        assert genre_sample_size <= n_genres, "You cannot set `genre_sample_size` greater than `n_genres`. The number" \
                                              "of sampled genres cannot exceed the number of available genres in" \
                                              "the dataset."
        self.movie_ratings = np.array(movie_ratings)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_users = n_users
        self.n_items = n_items
        self.n_genres = n_genres
        self.genre_sample_size = genre_sample_size

    def __len__(self):
        return int(np.ceil(self.movie_ratings.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.movie_ratings.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.movie_ratings[idxlist[start_idx:end_idx]]
            # get random user-item pairs - the probability to sample seen pairs, namely pairs in the dataset, it is
            # really low due to the sparsity of the dataset
            # even if we have a small number of sampled seen pairs it does not matter since that pairs act as a kind
            # of regularization for the target ratings, in fact they could correct wrong ground truth
            # this is the most efficient way to sample unseen pairs
            u_idx = np.random.choice(self.n_users, size=self.batch_size)
            i_idx = np.random.choice(self.n_items, size=self.batch_size)

            # note that the constant is used to represent the negative rating (rating equal to zero)
            # this is used by the Sim predicate

            yield (ltn.Variable('users_phi1', torch.tensor(data[:, 0]), add_batch_dim=False),
                   ltn.Variable('items_phi1', torch.tensor(data[:, 1]), add_batch_dim=False),
                   ltn.Variable('ratings', torch.tensor(data[:, -1]), add_batch_dim=False)), \
                  (ltn.Variable('users_phi2', torch.tensor(u_idx), add_batch_dim=False),
                   ltn.Variable('items_phi2', torch.tensor(i_idx), add_batch_dim=False),
                   # note that here we add self.n_items because we want to model genres after the movies in the MF model
                   ltn.Variable('genres', torch.randint(0, self.n_genres,
                                                        size=(self.genre_sample_size,)) + self.n_items,
                                add_batch_dim=False),
                   ltn.Constant(torch.tensor(0.)))


class TrainingDataLoaderLTNGenresNew:
    """
    Same loader as the previous one but since the genres are now separated from the movies, we do not need to add
    n_items when we sample the genres. See LTN variable genres on the previous loader and this loader to understand.
    """
    def __init__(self,
                 movie_ratings,
                 n_users,
                 n_items,
                 n_genres,
                 genre_sample_size=5,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the training data loader.
        :param movie_ratings: list of triples (user, item, rating)
        :param n_users: number of users in the dataset
        :param n_items: number of items in the dataset - this cannot be computed from training ratings since the
        procedure which creates the folds could have moved the only rating of one item in test set
        :param n_genres: number of movie genres in the dataset
        :param genre_sample_size: number of movie genres that have to be sampled at each batch (we sample to increase
        efficiency)
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        assert genre_sample_size <= n_genres, "You cannot set `genre_sample_size` greater than `n_genres`. The number" \
                                              "of sampled genres cannot exceed the number of available genres in" \
                                              "the dataset."
        self.movie_ratings = np.array(movie_ratings)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_users = n_users
        self.n_items = n_items
        self.n_genres = n_genres
        self.genre_sample_size = genre_sample_size

    def __len__(self):
        return int(np.ceil(self.movie_ratings.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.movie_ratings.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.movie_ratings[idxlist[start_idx:end_idx]]
            # get random user-item pairs - the probability to sample seen pairs, namely pairs in the dataset, it is
            # really low due to the sparsity of the dataset
            # even if we have a small number of sampled seen pairs it does not matter since that pairs act as a kind
            # of regularization for the target ratings, in fact they could correct wrong ground truth
            # this is the most efficient way to sample unseen pairs
            u_idx = np.random.choice(self.n_users, size=self.batch_size)
            i_idx = np.random.choice(self.n_items, size=self.batch_size)
            # todo see what is faster between random sample and np random Generator

            # note that the constant is used to represent the negative rating (rating equal to zero)
            # this is used by the Sim predicate

            yield (ltn.Variable('users_phi1', torch.tensor(data[:, 0]), add_batch_dim=False),
                   ltn.Variable('items_phi1', torch.tensor(data[:, 1]), add_batch_dim=False),
                   ltn.Variable('ratings', torch.tensor(data[:, -1]), add_batch_dim=False)), \
                  (ltn.Variable('users_phi2', torch.tensor(u_idx), add_batch_dim=False),
                   ltn.Variable('items_phi2', torch.tensor(i_idx), add_batch_dim=False),
                   # note that here we add self.n_items because we want to model genres after the movies in the MF model
                   ltn.Variable('genres', torch.randint(0, self.n_genres, size=(self.genre_sample_size,)),
                                add_batch_dim=False),
                   ltn.Constant(torch.tensor(-1. if 0 not in self.movie_ratings[:, -1] else 0.)))


class TrainingDataLoader:
    """
    Data loader to load the training set of the MindReader dataset. It creates batches composed of user-item pairs
    and their corresponding ratings.
    """

    def __init__(self,
                 data,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the training data loader.
        :param data: list of triples (user, item, rating)
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        self.data = np.array(data)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.data[idxlist[start_idx:end_idx]]
            u_i_pairs = data[:, :2]
            ratings = data[:, -1]

            yield torch.tensor(u_i_pairs), torch.tensor(ratings).float()


class TrainingDataLoaderImplicit:
    """
    Same loader as the previous one, but designed to work with implicit feedback. The data passed to the loader is just
    positive feedback. We need to sample a non-relevant item for each positive interaction. In this way, we provide the
    model with negative feedback as well.
    """

    def __init__(self,
                 data,
                 u_i_matrix,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the training data loader.
        :param data: list of triples (user, item, rating)
        :param u_i_matrix: sparse user-item matrix, used to sample the negative ratings (we do not want to sample some
        positive ratings at random)
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        self.data = np.array(data)
        self.u_i_matrix = u_i_matrix
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.data[idxlist[start_idx:end_idx]]
            negative_ints = []
            # here, we need to sample a non-relevant interaction for each positive interaction in the batch
            for user, _, _ in data:
                negative_ints.append([user, random.choice(list(set(range(self.u_i_matrix.shape[0])) -
                                                          set(self.u_i_matrix[user].nonzero()[1]))), 0])
            # convert negative interactions to numpy
            negative_ints = np.array(negative_ints)
            # add negative interactions to batch data
            data = np.concatenate([data, negative_ints], axis=0)
            u_i_pairs = data[:, :2]
            ratings = data[:, -1]

            yield torch.tensor(u_i_pairs), torch.tensor(ratings).float()


class TrainingDataLoaderBPRCustom:
    """
    Data loader designed to provide batches for learning a MF model with Bayesian Personalized Ranking. It is custom in
    the sense that it is designed to work with explicit feedback. The prepare_data method prepares the dataset for
    learning a model with BPR using both positive and negative interactions. For each user, for each positive
    interaction, this loaders matches the positive interaction with all the negative interactions of the user.
    """

    def __init__(self,
                 data,
                 u_i_matrix,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the training data loader.
        :param data: list of triples (user, item, rating)
        :param u_i_matrix: sparse user-item interaction matrix with 1 if user has interacted the item and 0 otherwise.
        It is used internally to sample non-relevant items for the training user that do not have negative items.
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        self.u_i_matrix = u_i_matrix
        self.data = self.prepare_data(data)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def prepare_data(self, data):
        data = pd.DataFrame(data, columns=['userId', 'uri', 'sentiment'])
        training_pairs = []
        users = data.groupby(["userId"])
        for user, user_ratings in users:
            pos_ratings = user_ratings[user_ratings["sentiment"] == 1]
            neg_ratings = user_ratings[user_ratings["sentiment"] != 1]
            # check that at least one example can be constructed for the current user
            if len(pos_ratings) >= 1 and len(neg_ratings) >= 1:
                training_pairs.extend([[[user, pos], [user, neg]]
                                       for pos, neg in list(itertools.product(pos_ratings["uri"],
                                                                              neg_ratings["uri"]))])
            else:
                # here, we sample one non-relevant item for each positive interaction of the user since we do not have
                # negative items for the current user
                # the probability of sampling a negative item is really high due to the high sparsity
                # very often, we can assume that non-relevant items are negative items
                # the problem is that they remain the same for all epochs, so there could be a bias on the selected
                # items. Performing experiments with more seeds is imperative in this case
                # todo there is a user for which we do not have examples because he has only negative examples in the
                #  training set. The validation on this user will be very challening but it is just a user
                if len(pos_ratings) >= 1 and len(neg_ratings) == 0:
                    training_pairs.extend([[[user, pos],
                                            [user, random.choice(list(set(range(self.u_i_matrix.shape[1])) -
                                             set(self.u_i_matrix[user].nonzero()[1])))]]
                                           for pos in list(pos_ratings["uri"])])
        return np.array(training_pairs)

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            yield torch.tensor(self.data[idxlist[start_idx:end_idx]])


class TrainingDataLoaderBPRClassic:
    """
    Data loader designed to provide batches for learning a MF model with Bayesian Personalized Ranking. This is the
    classic loader for learning a model using BPR. For each positive user-item interaction, one negative interaction
    is sampled from the set of non-relevant items for the user.
    """

    def __init__(self,
                 data,
                 u_i_matrix,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the training data loader.
        :param data: list of triples (user, item, rating)
        :param u_i_matrix: sparse user-item interaction matrix with 1 if user has interacted the item and 0 otherwise.
        It is used internally to sample non-relevant items for the training user that do not have negative items.
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        self.u_i_matrix = u_i_matrix
        # remove the rating column because it is not useful for this loader
        self.data = data[:, 0:2]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            positive_int = self.data[idxlist[start_idx:end_idx]]
            negative_int = []
            for user, _ in positive_int:
                negative_int.append([user, random.choice(list(set(range(self.u_i_matrix.shape[1])) -
                                                              set(self.u_i_matrix[user].nonzero()[1])))])
            negative_int = np.array(negative_int)
            yield torch.tensor(np.stack([positive_int, negative_int], axis=1))


class BalancedTrainingDataLoader:
    """
    A balanced version of the previous data loader. At each step, the same number of negative and positive interactions
    are sampled. During one epoch, the same negative interactions will be seen multiple times.
    """

    def __init__(self,
                 data,
                 batch_size=1,
                 sample_size=0.2,
                 shuffle=True):
        """
        Constructor of the training data loader.
        :param data: list of triples (user, item, rating)
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        data = np.array(data)
        self.positives = data[data[:, -1] == 1]
        self.negatives = data[data[:, -1] != 1]
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.positives.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.positives.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            positives = self.positives[idxlist[start_idx:end_idx]]
            negative_idx = np.random.randint(0, len(self.negatives), np.ceil(self.sample_size * len(positives)).astype(np.int64))
            negatives = self.negatives[negative_idx]
            data = np.concatenate((positives, negatives))
            u_i_pairs = data[:, :2]
            ratings = data[:, -1]

            yield torch.tensor(u_i_pairs), torch.tensor(ratings).float()


class BalancedTrainingDataLoaderNeg:
    """
    A balanced version of the previous data loader. At each step, the same number of negative and positive interactions
    are sampled. During one epoch, the same negative interactions will be seen multiple times.
    """

    def __init__(self,
                 data,
                 batch_size=1,
                 sample_size=0.2,
                 shuffle=True):
        """
        Constructor of the training data loader.
        :param data: list of triples (user, item, rating)
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        data = np.array(data)
        self.positives = data[data[:, -1] == 1]
        self.negatives = data[data[:, -1] != 1]
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.negatives.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.negatives.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            negatives = self.negatives[idxlist[start_idx:end_idx]]
            positive_idx = np.random.randint(0, len(self.positives), np.ceil(self.sample_size * len(negatives)).astype(np.int64))
            positives = self.positives[positive_idx]
            data = np.concatenate((positives, negatives))
            u_i_pairs = data[:, :2]
            ratings = data[:, -1]

            yield torch.tensor(u_i_pairs), torch.tensor(ratings).float()


class ValDataLoaderRanking:
    """
    Data loader to load the validation/test set of the MindReader dataset.
    """

    def __init__(self,
                 data,
                 batch_size=1):
        """
        Constructor of the validation data loader.
        :param data: matrix of user-item pairs. Every row is a user, where the last position contains the positive
        user-item pair, while the first 100 positions contain the negative user-item pairs
        :param batch_size: batch size for the validation/test of the model
        """
        self.data = np.array(data)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.data[idxlist[start_idx:end_idx]]
            ground_truth = np.zeros((data.shape[0], data.shape[1]))
            ground_truth[:, -1] = 1

            yield torch.tensor(data), ground_truth


class ValDataLoaderExact:
    """
    Data loader to load the validation/test set of the MindReader dataset for measuring exact metrics (all the items
    are ranked instead of sampling some of them).
    """

    def __init__(self,
                 te_u_ids,
                 tr_ui_matrix,
                 te_ui_matrix,
                 batch_size=1):
        """
        Constructor of the validation data loader.
        :param data: matrix of user-item pairs. Every row is a user, where the last position contains the positive
        user-item pair, while the first 100 positions contain the negative user-item pairs
        :param batch_size: batch size for the validation/test of the model
        """
        self.te_u_ids = te_u_ids
        self.tr_ui_matrix = tr_ui_matrix
        self.n_users = tr_ui_matrix.shape[0]
        self.n_items = tr_ui_matrix.shape[1]
        self.te_ui_matrix = te_ui_matrix
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.te_u_ids) / self.batch_size))

    def __iter__(self):
        n = len(self.te_u_ids)
        idxlist = list(range(n))

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            # to_predict = [[[u, i] for i in range(self.n_items)] for u in self.te_u_ids[idxlist[start_idx:end_idx]]]
            ground_truth = self.te_ui_matrix[self.te_u_ids[idxlist[start_idx:end_idx]]].toarray()
            mask = self.tr_ui_matrix[self.te_u_ids[idxlist[start_idx:end_idx]]].toarray()

            yield self.te_u_ids[idxlist[start_idx:end_idx]], mask, ground_truth


class ValDataLoaderRatings:
    """
    Data loader to load the validation/test set of the MindReader dataset for measuring MSE of the validation ratings.
    """

    def __init__(self,
                 data,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the training data loader.
        :param data: list of triples (user, item, rating)
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        self.data = np.array(data)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.data[idxlist[start_idx:end_idx]]
            u_i_pairs = data[:, :2]
            ratings = data[:, -1]

            yield torch.tensor(u_i_pairs), np.array(ratings).astype(np.float64)
