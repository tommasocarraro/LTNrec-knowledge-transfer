import os.path
import ltn
import torch
import numpy as np
from ltnrec.metrics import compute_metric, check_metrics
from ltnrec.loaders import ValDataLoader, TrainingDataLoader, TrainingDataLoaderLTN, TrainingDataLoaderLTNGenres
import json
from torch.optim import Adam
from ltnrec.utils import append_to_result_file, set_seed
import wandb


class MatrixFactorization(torch.nn.Module):
    """
    Matrix factorization model.
    The model has inside two matrices: one containing the embeddings of the users of the system, one containing the
    embeddings of the items of the system.
    """
    def __init__(self, n_users, n_items, n_factors, biased=False):
        """
        Construction of the matrix factorization model.
        :param n_users: number of users in the dataset
        :param n_items: number of items in the dataset
        :param n_factors: size of embeddings for users and items
        :param biased: whether the MF model must include user and item biases or not, default to False
        """
        super(MatrixFactorization, self).__init__()
        self.u_emb = torch.nn.Embedding(n_users, n_factors)
        self.i_emb = torch.nn.Embedding(n_items, n_factors)
        torch.nn.init.xavier_normal_(self.u_emb.weight)
        torch.nn.init.xavier_normal_(self.i_emb.weight)
        self.biased = biased
        if biased:
            self.u_bias = torch.nn.Embedding(n_users, 1)
            self.i_bias = torch.nn.Embedding(n_items, 1)
            torch.nn.init.normal_(self.u_bias.weight)
            torch.nn.init.normal_(self.i_bias.weight)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, u_idx, i_idx, dim=1, normalize=False):
        """
        It computes the scores for the given user-item pairs using the matrix factorization approach (dot product).
        :param u_idx: users for which the score has to be computed
        :param i_idx: items for which the score has to be computed
        :param dim: dimension along which the dot product has to be computed
        :param normalize: whether the output must be normalized in [0., 1.] (e.g., for a predicate) or not (logits)
        :return: predicted scores for given user-item pairs
        """
        pred = torch.sum(self.u_emb(u_idx) * self.i_emb(i_idx), dim=dim, keepdim=True)
        if self.biased:
            pred += self.u_bias(u_idx) + self.i_bias(i_idx)
        return torch.sigmoid(pred.squeeze()) if normalize else pred.squeeze()


class Trainer:
    """
    Abstract base class that any trainer must inherit from.
    """
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train(self, train_loader, val_loader, val_metric, n_epochs=200, early=None, verbose=10, save_path=None,
              model_name=None, wandb_train=False):
        """
        Method for the train of the model.
        :param train_loader: data loader for training data
        :param val_loader: data loader for validation data
        :param val_metric: validation metric name
        :param n_epochs: number of epochs of training, default to 200
        :param early: threshold for early stopping, default to None
        :param verbose: number of epochs to wait for printing training details (every 'verbose' epochs)
        :param save_path: path where to save the best model, default to None
        :param model_name: name of model. It is used for printing model information in the log. It is useful when
        concurrency is used to train simultaneously.
        :param wandb_train: whether the training is done on Weights and Biases or not.
        """
        best_val_score = 0.0
        early_counter = 0
        check_metrics(val_metric)

        for epoch in range(n_epochs):
            # training step
            train_loss = self.train_epoch(train_loader)
            # validation step
            val_score = self.validate(val_loader, val_metric)
            # print epoch data
            if (epoch + 1) % verbose == 0:
                log_record = "Epoch %d - Train loss %.3f - Validation %s %.3f" % (epoch + 1, train_loss, val_metric,
                                                                                  val_score)
                if model_name is not None:
                    log_record = ("%s : " % model_name) + log_record
                if wandb_train:
                    wandb.log({
                        'val_%s' % (val_metric, ): val_score
                    })
                print(log_record)
            # save best model and update early stop counter, if necessary
            if val_score > best_val_score:
                best_val_score = val_score
                early_counter = 0
                if save_path:
                    self.save_model(save_path)
            else:
                early_counter += 1
                if early is not None and early_counter > early:
                    print("Training interrupted due to early stopping")
                    break

    def train_epoch(self, train_loader):
        """
        Method for the training of one single epoch.
        :param train_loader: data loader for training data
        :return: training loss value averaged across training batches
        """
        raise NotImplementedError()

    def predict(self, x, *args, **kwargs):
        """
        Method for performing a prediction of the model.
        :param x: input for which the prediction has to be performed
        :param args: these are the potential additional parameters useful to the model for performing the prediction
        :param kwargs: these are the potential additional parameters useful to the model for performing the prediction
        :return: prediction of the model for the given input
        """

    def validate(self, val_loader, val_metric):
        """
        Method for validating the model.
        :param val_loader: data loader for validation data
        :param val_metric: validation metric name
        :return: validation score based on the given metric averaged across validation batches
        """
        raise NotImplementedError()

    def save_model(self, path):
        """
        Method for saving the model.
        :param path: path where to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        """
        Method for loading the model.
        :param path: path from which the model has to be loaded.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def test(self, test_loader, metrics):
        """
        Method for performing the test of the model based on the given test data and test metrics.
        :param test_loader: data loader for test data
        :param metrics: metric name or list of metrics' names that have to be computed
        :return: a dictionary containing the value of each metric average across the test batches
        """
        raise NotImplementedError()


class MFTrainer(Trainer):
    """
    Trainer for the Matrix Factorization model.
    The objective of the Matrix Factorization is to minimize the MSE between the predictions of the model and the
    ground truth.
    """
    def __init__(self, mf_model, optimizer):
        """
        Constructor of the trainer for the MF model.
        :param mf_model: Matrix Factorization model
        :param optimizer: optimizer used for the training of the model
        """
        super(MFTrainer, self).__init__(mf_model, optimizer)
        self.mse = torch.nn.MSELoss()

    def train_epoch(self, train_loader):
        train_loss = 0.0
        for batch_idx, (u_i_pairs, ratings) in enumerate(train_loader):
            self.optimizer.zero_grad()
            loss = self.mse(self.model(u_i_pairs[:, 0], u_i_pairs[:, 1]), ratings)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_loader)

    def predict(self, x, dim=1, normalize=False):
        """
        Method for performing a prediction of the model.
        :param x: tensor containing the user-item pair for which the prediction has to be computed. The first position
        is the user index, while the second position is the item index
        :param dim: dimension across which the dot product of the MF has to be computed
        :param normalize: whether to normalize the prediction in the range [0., 1.]
        :return: the prediction of the model for the given user-item pair
        """
        u_idx, i_idx = x[:, 0], x[:, 1]
        with torch.no_grad():
            return self.model(u_idx, i_idx, dim, normalize)

    def validate(self, val_loader, val_metric):
        val_score = []
        for batch_idx, (data, ground_truth) in enumerate(val_loader):
            predicted_scores = self.predict(data.view(-1, 2))
            val_score.append(compute_metric(val_metric, predicted_scores.view(ground_truth.shape).numpy(),
                                            ground_truth))
        return np.mean(np.concatenate(val_score))

    def test(self, test_loader, metrics):
        check_metrics(metrics)
        if isinstance(metrics, str):
            metrics = [metrics]

        results = {m: [] for m in metrics}
        for batch_idx, (data, ground_truth) in enumerate(test_loader):
            for m in results:
                predicted_scores = self.predict(data.view(-1, 2))
                results[m].append(compute_metric(m, predicted_scores.view(ground_truth.shape).numpy(), ground_truth))
        for m in results:
            results[m] = np.mean(np.concatenate(results[m]))
        return results


class LTNTrainerMF(MFTrainer):
    """
    Trainer for the Logic Tensor Network with Matrix Factorization as the predictive model for the Likes function.
    The Likes function takes as input a user-item pair and produce an un-normalized score (MF). Ideally, this score
    should be near 1 if the user likes the item, and near 0 if the user dislikes the item.
    The closeness between the predictions of the Likes function and the ground truth provided by the dataset for the
    training user-item pairs is obtained by maximizing the truth value of the predicate Sim. The predicate Sim takes
    as input a predicted score and the ground truth, and returns a value in the range [0., 1.]. Higher the value,
    higher the closeness between the predicted score and the ground truth.
    """
    def __init__(self, mf_model, optimizer, alpha, p=2):
        """
        Constructor of the trainer for the LTN with MF as base model.
        :param mf_model: Matrix Factorization model to implement the Likes function
        :param optimizer: optimizer used for the training of the model
        :param alpha: coefficient of smooth equality predicate
        :param p: hyper-parameter p for pMeanError of the quantifier
        """
        super(LTNTrainerMF, self).__init__(mf_model, optimizer)
        self.Likes = ltn.Function(self.model)
        self.Sim = ltn.Predicate(func=lambda pred, gt: torch.exp(-alpha * torch.square(pred - gt)))
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=p), quantifier='f')
        self.sat_agg = ltn.fuzzy_ops.SatAgg()

    def train_epoch(self, train_loader):
        train_loss = 0.0
        for batch_idx, (u, i, r) in enumerate(train_loader):
            self.optimizer.zero_grad()
            train_sat = self.Forall(ltn.diag(u, i, r), self.Sim(self.Likes(u, i), r)).value
            loss = 1. - train_sat
            loss.backward()
            self.optimizer.step()
            train_loss += train_sat.item()

        return train_loss / len(train_loader)


class LTNTrainerMFGenres(LTNTrainerMF):
    """
    Trainer for the Logic Tensor Network with Matrix Factorization as the predictive model for the Likes function. In
    addition, unlike the previous model, this LTN has an additional axiom in the loss function. This axiom serves as
    a kind of regularization for the embeddings learned by the MF model. The axiom states that if a user dislikes a
    movie genre, then if a movie has that genre, the user should dislike it.
    The Likes function takes as input a user-item pair and produce an un-normalized score (MF). Ideally, this score
    should be near 1 if the user likes the item, and near 0 if the user dislikes the item.
    The closeness between the predictions of the Likes function and the ground truth provided by the dataset for the
    training user-item pairs is obtained by maximizing the truth value of the predicate Sim. The predicate Sim takes
    as input a predicted score and the ground truth, and returns a value in the range [0., 1.]. Higher the value,
    higher the closeness between the predicted score and the ground truth.
    """

    def __init__(self, mf_model, optimizer, alpha, p, n_movies, item_genres_matrix):
        """
        Constructor of the trainer for the LTN with MF as base model.
        :param mf_model: Matrix Factorization model to implement the Likes function
        :param optimizer: optimizer used for the training of the model
        :param alpha: coefficient of smooth equality predicate
        :param p: hyper-parameter p for pMeanError of rule on genres
        :param n_movies: number of movies in the dataset
        :param item_genres_matrix: sparse matrix with items on the rows and genres on the columns. A 1 means the item
        belongs to the genre
        """
        super(LTNTrainerMFGenres, self).__init__(mf_model, optimizer, alpha, p)
        item_genres_matrix = torch.tensor(item_genres_matrix.todense())
        self.Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(), quantifier="e")
        self.And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        # here, we need to remove n_movies because the genres are in [n_movies, n_movies + n_genres] in the MF model
        # instead, in the item_genres_matrix they are in [0, n_genres]
        self.HasGenre = ltn.Predicate(func=lambda i_idx, g_idx: item_genres_matrix[i_idx, g_idx - n_movies])

    def train_epoch(self, train_loader):
        train_loss = 0.0
        for batch_idx, ((u1, i1, r), (u2, i2, g, r_)) in enumerate(train_loader):
            self.optimizer.zero_grad()
            f1 = self.Forall(ltn.diag(u1, i1, r), self.Sim(self.Likes(u1, i1), r))
            f2 = self.Forall(ltn.diag(u2, i2),
                             self.Forall(g, self.Implies(
                                 self.And(self.Sim(self.Likes(u2, g), r_), self.HasGenre(i2, g)),
                                 self.Sim(self.Likes(u2, i2), r_))))
            train_sat = self.sat_agg(f1, f2)
            loss = 1. - train_sat
            loss.backward()
            self.optimizer.step()
            train_loss += train_sat.item()

        return train_loss / len(train_loader)


class Model:
    """
    Abstract base class that any model must inherit from.
    """
    def __init__(self, config_file_name):
        """
        It constructs a model.
        :param config_file_name: name of the configuration file containing the hyper-parameters of the model and the
        values that have to be tried during the grid search.
        """
        self.model_name = config_file_name
        assert os.path.exists("./config/%s.json" % self.model_name), "The configuration file specified " \
                                                                     "does not exist. Please, add " \
                                                                     "it at ./config"
        with open("./config/%s.json" % self.model_name) as json_file:
            self.param_config = json.load(json_file)

    def grid_search(self, data, seed, save_path_prefix):
        """
        It performs a grid search of the model, with hyper-parameters specified in the corresponding config file.
        It creates a JSON file containing the best configuration of hyper-parameters.
        prefix of the path where to save the weights of the best model. It is also the prefix
        of the path where to save the best hyper-parameters configuration.
        """
        pass

    def run_experiment(self, data, seed, save_path_prefix, result_file_name):
        """
        It trains and tests the model. To train the model, it uses the hyper-parameters specified in the config file
        created by the grid_search() method. If this file does not exist, it runs a grid search to find the best
        hyper-parameters and then trains the model with the best hyper-parameters.
        """
        pass


class StandardMFModel(Model):
    """
    Standard matrix factorization model. It uses the MSE between target and predicted ratings as an objective.
    """
    def __init__(self, config_file_name="standard_mf"):
        super(StandardMFModel, self).__init__(config_file_name)

    def grid_search(self, data, seed, save_path_prefix):
        if not os.path.exists("./config/best_config/%s-%s.json" % (save_path_prefix, self.model_name)):
            set_seed(seed)
            best_score = 0.0
            val_loader = ValDataLoader(data.val, self.param_config['val_batch_size'])
            print("Starting grid search of model %s on %s" % (self.model_name, save_path_prefix))
            for k in self.param_config["k"]:
                for biased in self.param_config["biased"]:
                    for lr in self.param_config["lr"]:
                        for wd in self.param_config["wd"]:
                            for tr_batch_size in self.param_config["tr_batch_size"]:
                                print("Training %s on %s with config: k=%d, biased=%d, lr=%.3f, wd=%.4f, batch_size=%d" %
                                      (self.model_name, save_path_prefix, k, biased, lr, wd, tr_batch_size))
                                # train model with current configuration
                                model = MatrixFactorization(data.n_users, data.n_items, k, biased)
                                tr_loader = TrainingDataLoader(data.train, tr_batch_size)
                                trainer = MFTrainer(model, Adam(model.parameters(), lr=lr, weight_decay=wd))
                                trainer.train(tr_loader, val_loader, self.param_config["val_metric"],
                                              n_epochs=self.param_config["n_epochs"], early=self.param_config["early_stop"],
                                              verbose=self.param_config["verbose"],
                                              save_path="./saved_models/temporary/%s-%s.pth" % (save_path_prefix,
                                                                                                self.model_name),
                                              model_name="grid_search:%s-%s" % (save_path_prefix, self.model_name))
                                # load best weights for current configuration
                                trainer.load_model("./saved_models/temporary/%s-%s.pth" % (save_path_prefix,
                                                                                           self.model_name))
                                # get validation score for best weights
                                val_score = trainer.test(val_loader,
                                                         self.param_config["val_metric"])[self.param_config["val_metric"]]
                                # check if it is the best score found
                                if val_score > best_score:
                                    # update best score
                                    best_score = val_score
                                    # save best configuration of parameters
                                    config_dict = {
                                        "k": k,
                                        "biased": biased,
                                        "lr": lr,
                                        "wd": wd,
                                        "tr_batch_size": tr_batch_size
                                    }
                                    with open("./config/best_config/%s-%s.json"
                                              % (save_path_prefix, self.model_name), "w") as outfile:
                                        json.dump(config_dict, outfile, indent=4)

    def run_experiment(self, data, seed, save_path_prefix, result_file_name):
        if not os.path.exists("./saved_models/seed_%d" % seed):
            os.mkdir("./saved_models/seed_%d" % seed)
        # check if a grid search is needed
        if not os.path.exists("./config/best_config/%s-%s.json" % (save_path_prefix, self.model_name)):
            # we compute the grid search
            self.grid_search(data, seed, save_path_prefix)
        # load best configuration of hyper-parameters
        with open("./config/best_config/%s-%s.json" % (save_path_prefix, self.model_name)) as json_file:
            best_config = json.load(json_file)
        set_seed(seed)
        # train the model with the best configuration
        model = MatrixFactorization(data.n_users, data.n_items, best_config['k'], best_config['biased'])
        tr_loader = TrainingDataLoader(data.train, best_config['tr_batch_size'])
        val_loader = ValDataLoader(data.val, batch_size=self.param_config['val_batch_size'])
        test_loader = ValDataLoader(data.test, batch_size=self.param_config['val_batch_size'])
        trainer = MFTrainer(model, Adam(model.parameters(), lr=best_config['lr'], weight_decay=best_config['wd']))
        # check if the model has been already trained before
        if not os.path.exists("./saved_models/seed_%d/%s-%s.pth" % (seed, save_path_prefix, self.model_name)):
            print("Starting training of %s on %s with seed %d" % (self.model_name, save_path_prefix, seed))
            trainer.train(tr_loader, val_loader, self.param_config["val_metric"],
                          n_epochs=self.param_config["n_epochs"], early=self.param_config["early_stop"],
                          verbose=self.param_config["verbose"],
                          save_path="./saved_models/seed_%d/%s-%s.pth" % (seed, save_path_prefix, self.model_name),
                          model_name="%s-%s-seed_%d" % (save_path_prefix, self.model_name, seed))
        # load best weights of the model trained with best config
        trainer.load_model("./saved_models/seed_%d/%s-%s.pth" % (seed, save_path_prefix, self.model_name))
        # test the model and save the results
        append_to_result_file(result_file_name, "%s-%s" % (save_path_prefix, self.model_name),
                              trainer.test(test_loader, self.param_config["test_metrics"]), seed)


class LTNMFModel(Model):
    """
    Matrix factorization model trained using LTN. It uses a logic formula that forces the target ratings to be as
    similar as possible to the predicted ratings as an objective.
    """
    def __init__(self, config_file_name="ltn_mf"):
        super(LTNMFModel, self).__init__(config_file_name)

    def grid_search(self, data, seed, save_path_prefix):
        if not os.path.exists("./config/best_config/%s-%s.json" % (save_path_prefix, self.model_name)):
            set_seed(seed)
            best_score = 0.0
            val_loader = ValDataLoader(data.val, self.param_config['val_batch_size'])
            print("Starting grid search of model %s on %s" % (self.model_name, save_path_prefix))
            for k in self.param_config["k"]:
                for biased in self.param_config["biased"]:
                    for lr in self.param_config["lr"]:
                        for wd in self.param_config["wd"]:
                            for tr_batch_size in self.param_config["tr_batch_size"]:
                                for alpha in self.param_config["alpha"]:
                                    print("Training %s on %s with config: k=%d, biased=%d, lr=%.3f, wd=%.4f, "
                                          "batch_size=%d, alpha=%.3f" % (self.model_name, save_path_prefix, k, biased,
                                                                         lr, wd, tr_batch_size, alpha))
                                    # train model with current configuration
                                    model = MatrixFactorization(data.n_users, data.n_items, k, biased)
                                    tr_loader = TrainingDataLoaderLTN(data.train, tr_batch_size)
                                    trainer = LTNTrainerMF(model, Adam(model.parameters(), lr=lr, weight_decay=wd), alpha)
                                    trainer.train(tr_loader, val_loader, self.param_config["val_metric"],
                                                  n_epochs=self.param_config["n_epochs"],
                                                  early=self.param_config["early_stop"],
                                                  verbose=self.param_config["verbose"],
                                                  save_path="./saved_models/temporary/%s-%s.pth" % (save_path_prefix,
                                                                                                    self.model_name),
                                                  model_name="grid_search:%s-%s" % (save_path_prefix, self.model_name))
                                    # load best weights for current configuration
                                    trainer.load_model("./saved_models/temporary/%s-%s.pth" % (save_path_prefix,
                                                                                               self.model_name))
                                    # get validation score for best weights
                                    val_score = trainer.test(val_loader,
                                                             self.param_config["val_metric"])[self.param_config["val_metric"]]
                                    # check if it is the best score found
                                    if val_score > best_score:
                                        # update best score
                                        best_score = val_score
                                        # save best configuration of parameters
                                        config_dict = {
                                            "k": k,
                                            "biased": biased,
                                            "lr": lr,
                                            "wd": wd,
                                            "tr_batch_size": tr_batch_size,
                                            "alpha": alpha
                                        }
                                        with open("./config/best_config/%s-%s.json"
                                                  % (save_path_prefix, self.model_name), "w") as outfile:
                                            json.dump(config_dict, outfile, indent=4)

    def run_experiment(self, data, seed, save_path_prefix, result_file_name):
        if not os.path.exists("./saved_models/seed_%d" % seed):
            os.mkdir("./saved_models/seed_%d" % seed)
        # check if a grid search is needed
        if not os.path.exists("./config/best_config/%s-%s.json" % (save_path_prefix, self.model_name)):
            # we compute the grid search
            self.grid_search(data, seed, save_path_prefix)
        # load best configuration of hyper-parameters
        with open("./config/best_config/%s-%s.json" % (save_path_prefix, self.model_name)) as json_file:
            best_config = json.load(json_file)
        set_seed(seed)
        # train the model with the best configuration
        model = MatrixFactorization(data.n_users, data.n_items, best_config['k'], best_config['biased'])
        tr_loader = TrainingDataLoaderLTN(data.train, best_config['tr_batch_size'])
        val_loader = ValDataLoader(data.val, batch_size=self.param_config['val_batch_size'])
        test_loader = ValDataLoader(data.test, batch_size=self.param_config['val_batch_size'])
        trainer = LTNTrainerMF(model, Adam(model.parameters(), lr=best_config['lr'], weight_decay=best_config['wd']),
                               best_config["alpha"])
        # check if the model has been already trained before
        if not os.path.exists("./saved_models/seed_%d/%s-%s.pth" % (seed, save_path_prefix, self.model_name)):
            print("Starting training of %s on %s with seed %d" % (self.model_name, save_path_prefix, seed))
            trainer.train(tr_loader, val_loader, self.param_config["val_metric"],
                          n_epochs=self.param_config["n_epochs"], early=self.param_config["early_stop"],
                          verbose=self.param_config["verbose"],
                          save_path="./saved_models/seed_%d/%s-%s.pth" % (seed, save_path_prefix, self.model_name),
                          model_name="%s-%s-seed_%d" % (save_path_prefix, self.model_name, seed))
        # load best weights of the model trained with best config
        trainer.load_model("./saved_models/seed_%d/%s-%s.pth" % (seed, save_path_prefix, self.model_name))
        # test the model and save the results
        append_to_result_file(result_file_name, "%s-%s" % (save_path_prefix, self.model_name),
                              trainer.test(test_loader, self.param_config["test_metrics"]), seed)


class LTNMFGenresModel(Model):
    """
    Matrix factorization model trained using LTN. It uses two logic formulas.
    One forces the target ratings to be as similar as possible to the predicted ratings.
    The other one performs logical reasoning on the relationships between users, movies, and genres.
    In particular, for the movies that have not been rated by a user, it forces the score to be low if the user did not
    like a genre of that movie in the past.
    """
    def __init__(self, config_file_name="ltn_mf_genres"):
        super(LTNMFGenresModel, self).__init__(config_file_name)

    def grid_search(self, data, seed, save_path_prefix):
        if not os.path.exists("./config/best_config/%s-%s.json" % (save_path_prefix, self.model_name)):
            set_seed(seed)
            best_score = 0.0
            val_loader = ValDataLoader(data.val, self.param_config['val_batch_size'])
            print("Starting grid search of model %s on %s" % (self.model_name, save_path_prefix))
            for k in self.param_config["k"]:
                for biased in self.param_config["biased"]:
                    for lr in self.param_config["lr"]:
                        for wd in self.param_config["wd"]:
                            for tr_batch_size in self.param_config["tr_batch_size"]:
                                for alpha in self.param_config["alpha"]:
                                    for p in self.param_config["p"]:
                                        print("Training %s on %s with config: k=%d, biased=%d, lr=%.3f, wd=%.4f, "
                                              "batch_size=%d, alpha=%.3f, p=%d" % (self.model_name, save_path_prefix, k,
                                                                                   biased, lr, wd, tr_batch_size, alpha, p))
                                        # train model with current configuration
                                        model = MatrixFactorization(data.n_users, data.n_items, k, biased)
                                        tr_loader = TrainingDataLoaderLTNGenres(data.train, data.n_users,
                                                                                data.n_items - data.n_genres,
                                                                                data.n_genres, genre_sample_size=5,
                                                                                batch_size=tr_batch_size)
                                        trainer = LTNTrainerMFGenres(model, Adam(model.parameters(), lr=lr,
                                                                                 weight_decay=wd), alpha, p,
                                                                     data.n_items - data.n_genres,
                                                                     data.item_genres_matrix)
                                        trainer.train(tr_loader, val_loader, self.param_config["val_metric"],
                                                      n_epochs=self.param_config["n_epochs"],
                                                      early=self.param_config["early_stop"],
                                                      verbose=self.param_config["verbose"],
                                                      save_path="./saved_models/temporary/%s-%s.pth" % (save_path_prefix,
                                                                                                        self.model_name),
                                                      model_name="grid_search:%s-%s" % (save_path_prefix,
                                                                                        self.model_name))
                                        # load best weights for current configuration
                                        trainer.load_model("./saved_models/temporary/%s-%s.pth" % (save_path_prefix,
                                                                                                   self.model_name))
                                        # get validation score for best weights
                                        val_score = trainer.test(val_loader,
                                                                 self.param_config["val_metric"])[self.param_config["val_metric"]]
                                        # check if it is the best score found
                                        if val_score > best_score:
                                            # update best score
                                            best_score = val_score
                                            # save best configuration of parameters
                                            config_dict = {
                                                "k": k,
                                                "biased": biased,
                                                "lr": lr,
                                                "wd": wd,
                                                "tr_batch_size": tr_batch_size,
                                                "alpha": alpha,
                                                "p": p
                                            }
                                            with open("./config/best_config/%s-%s.json"
                                                      % (save_path_prefix, self.model_name), "w") as outfile:
                                                json.dump(config_dict, outfile, indent=4)

    def run_experiment(self, data, seed, save_path_prefix, result_file_name):
        if not os.path.exists("./saved_models/seed_%d" % seed):
            os.mkdir("./saved_models/seed_%d" % seed)
        # check if a grid search is needed
        if not os.path.exists("./config/best_config/%s-%s.json" % (save_path_prefix, self.model_name)):
            # we compute the grid search
            self.grid_search(data, seed, save_path_prefix)
        # load best configuration of hyper-parameters
        with open("./config/best_config/%s-%s.json" % (save_path_prefix, self.model_name)) as json_file:
            best_config = json.load(json_file)
        set_seed(seed)
        # train the model with the best configuration
        model = MatrixFactorization(data.n_users, data.n_items, best_config['k'], best_config['biased'])
        tr_loader = TrainingDataLoaderLTNGenres(data.train, data.n_users, data.n_items - data.n_genres, data.n_genres,
                                                genre_sample_size=5, batch_size=best_config["tr_batch_size"])
        val_loader = ValDataLoader(data.val, batch_size=self.param_config['val_batch_size'])
        test_loader = ValDataLoader(data.test, batch_size=self.param_config['val_batch_size'])
        trainer = LTNTrainerMFGenres(model, Adam(model.parameters(), lr=best_config['lr'],
                                                 weight_decay=best_config['wd']), best_config["alpha"], best_config["p"],
                                     data.n_items - data.n_genres, data.item_genres_matrix)
        # check if the model has been already trained before
        if not os.path.exists("./saved_models/seed_%d/%s-%s.pth" % (seed, save_path_prefix, self.model_name)):
            print("Starting training of %s on %s with seed %d" % (self.model_name, save_path_prefix, seed))
            trainer.train(tr_loader, val_loader, self.param_config["val_metric"],
                          n_epochs=self.param_config["n_epochs"], early=self.param_config["early_stop"],
                          verbose=self.param_config["verbose"],
                          save_path="./saved_models/seed_%d/%s-%s.pth" % (seed, save_path_prefix, self.model_name),
                          model_name="%s-%s-seed_%d" % (save_path_prefix, self.model_name, seed))
        # load best weights of the model trained with best config
        trainer.load_model("./saved_models/seed_%d/%s-%s.pth" % (seed, save_path_prefix, self.model_name))
        # test the model and save the results
        append_to_result_file(result_file_name, "%s-%s" % (save_path_prefix, self.model_name),
                              trainer.test(test_loader, self.param_config["test_metrics"]), seed)
