import numpy as np
import torch
import ltn
from scipy.sparse import csr_matrix
# todo fare un loader che fornisce anche i rating sui generi
# todo fare matrix factorization tra user e genre, basta aggiungere i fattori latenti dei generi in coda a
#  quelli dei film -> i generi sono diversi dai film, quindi non capisco perche' dovrebbero stare assieme
# todo usare one-hot per i generi e usare una rete che predice lo score
# todo creare un val set anche per i generi
# todo aggiungere la formula 2 e vedere se porta a miglioramenti, sopratutto sui casi di cold start
# todo provare a rendere piu' sparso ml-100k e far vedere che i rating di mindreader aiutano


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


# per ogni utente del batch, vedo quali generi non gli piacciono, poi vedo quali di questi generi compaiono nei film
# del batch per ogni utente
# mi costruisco un dataset di tuple in cui ho coppie utente item che non compaiono nel dataset
# faccio sample da queste
# guardo se l'item ha almeno un genere che all'utente non piace e in quel caso diminuisco il likes
class TrainingDataLoaderLTNGenres:
    """
    Data loader to load the training set of the dataset. It creates batches and wrap them inside LTN
    variables ready for the learning. This loaders differs from the TrainingDataLoaderLTN. In particular, it creates
    LTN variables to reason on formulas which involve the genres of the movies.
    """

    def __init__(self,
                 movie_ratings,
                 n_users,
                 n_items,
                 n_genres,
                 movie_genres,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the training data loader.
        :param movie_ratings: list of triples (user, item, rating)
        :param n_users: number of users in the dataset
        :param n_items: number of items in the dataset - this cannot be computed from training ratings since the
        procedure which creates the folds could have moved the only rating of one item in test set
        :param n_genres: number of movie genres in the dataset
        :param movie_genres: dictionary containing for each movie the genres to which it belongs to
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        self.movie_ratings = np.array(movie_ratings)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_users = n_users
        self.n_items = n_items
        self.n_genres = n_genres
        self.movie_genres = self.get_movie_genre_matrix(movie_genres)

    def get_movie_genre_matrix(self, movie_genres):
        movie_genre_pairs = np.array([[movie, int(genre)] for movie in movie_genres
                                      for genre in movie_genres[movie]
                                      if genre != 'None'])
        return csr_matrix((np.ones(len(movie_genre_pairs)), (movie_genre_pairs[:, 0], movie_genre_pairs[:, 1])),
                          shape=(self.n_items, self.n_genres))

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
            # really low due to sparsity of dataset
            # even if we have a small number of sampled seen pairs, it does not matter
            # this is the most efficient way to sample unseen pairs
            u_idx = np.random.choice(self.n_users, size=self.batch_size)
            i_idx = np.random.choice(self.n_items, size=self.batch_size)
            unseen_pairs = np.concatenate([u_idx[:, np.newaxis], i_idx[:, np.newaxis]], axis=1)
            item_to_genre = self.movie_genres[i_idx]
            # remove seen pairs from randomly sampled pairs
            # this is not efficient, hence removed
            # unseen_pairs = np.array([pair for pair in unseen_pairs if pair not in self.movie_ratings_list])

            yield (ltn.Variable('users', torch.tensor(data[:, 0]), add_batch_dim=False),
                   ltn.Variable('items', torch.tensor(data[:, 1]), add_batch_dim=False),
                   ltn.Variable('ratings', torch.tensor(data[:, -1]), add_batch_dim=False))


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
            ratings[ratings == -1] = 0

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
