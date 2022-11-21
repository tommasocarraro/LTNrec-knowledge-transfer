import numpy as np
import torch
import ltn
# todo usare one-hot per i generi e usare una rete che predice lo score
# todo creare un val set anche per i generi
# todo aggiungere la formula 2 e vedere se porta a miglioramenti, soprattutto sui casi di cold start
# todo provare a rendere piu' sparso ml-100k e far vedere che i rating di mindreader aiutano -> interessante (stesso
#  studio che abbiamo fatto sull'altro paper)


class TrainingDataLoaderLTN:
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
                   ltn.Variable('genres', torch.randint(0, self.n_genres,
                                                        size=(self.genre_sample_size,)) + self.n_items,
                                add_batch_dim=False),
                   ltn.Constant(torch.tensor(0.)))


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


class ValDataLoader:
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
