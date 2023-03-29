import os.path
import sys
import ltn
import torch
import numpy as np
import torch.nn.functional as F
from ltnrec.metrics import compute_metric, check_metrics
from ltnrec.loaders import ValDataLoaderRatings, TrainingDataLoader, TrainingDataLoaderLTNGenres, ValDataLoaderRanking
import json
from torch.optim import Adam
from ltnrec.utils import set_seed, remove_seed_from_dataset_name
import wandb
import pickle
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

# create global wandb api object
api = wandb.Api()
api.entity = "bmxitalia"


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.2, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = torch.nn.BCELoss()
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        loss = torch.where(targets == 1,
                           self.alpha * ((1 - inputs) ** self.gamma) * bce_loss,
                           (1 - self.alpha) * (inputs ** self.gamma) * bce_loss)
        return torch.mean(loss) if self.reduction == "mean" else torch.sum(loss)


class FocalLossPyTorch(torch.nn.Module):
    def __init__(self, alpha=0.2, gamma=2, reduction="mean"):
        super(FocalLossPyTorch, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = torch.nn.BCELoss(reduction="none")
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: "
                f"'none', 'mean', 'sum'")
        return loss


class MatrixFactorization(torch.nn.Module):
    """
    Matrix factorization model.
    The model has inside two matrices: one containing the embeddings of the users of the system, one containing the
    embeddings of the items of the system.
    """

    def __init__(self, n_users, n_items, n_factors, biased=False, init_std=0.001):
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
        torch.nn.init.normal_(self.u_emb.weight, 0.0, init_std)
        torch.nn.init.normal_(self.i_emb.weight, 0.0, init_std)
        self.biased = biased
        if biased:
            self.u_bias = torch.nn.Embedding(n_users, 1)
            self.i_bias = torch.nn.Embedding(n_items, 1)
            torch.nn.init.normal_(self.u_bias.weight, 0.0, init_std)
            torch.nn.init.normal_(self.i_bias.weight, 0.0, init_std)

    def forward(self, u_idx, i_idx, dim=1):
        """
        It computes the scores for the given user-item pairs using the matrix factorization approach (dot product).
        :param u_idx: users for which the score has to be computed
        :param i_idx: items for which the score has to be computed
        :param dim: dimension along which the dot product has to be computed
        :return: predicted scores for given user-item pairs
        """
        pred = torch.sum(self.u_emb(u_idx) * self.i_emb(i_idx), dim=dim, keepdim=True)
        if self.biased:
            pred += self.u_bias(u_idx) + self.i_bias(i_idx)
        return pred.squeeze()


class MatrixFactorizationLTN(MatrixFactorization):
    """
    Matrix factorization model designed to work with the LTN framework.
    The model has inside two matrices: one containing the embeddings of the users of the system, one containing the
    embeddings of the items of the system.
    """

    def __init__(self, n_users, n_items, n_factors, biased=False, init_std=0.001, normalize=False):
        """
        Construction of the matrix factorization model.
        :param n_users: number of users in the dataset
        :param n_items: number of items in the dataset
        :param n_factors: size of embeddings for users and items
        :param biased: whether the MF model must include user and item biases or not, default to False
        :param normalize: whether the output has to be normalized in [0., 1.] or not. It is useful when this model
        is used with the LTN framework. In fact, this model could implement an LTN function or predicate. Predicates
        require the output to be in the range [0., 1.] while functions usually not. Defaults to False.
        """
        super(MatrixFactorization, self).__init__()
        self.u_emb = torch.nn.Embedding(n_users, n_factors)
        self.i_emb = torch.nn.Embedding(n_items, n_factors)
        torch.nn.init.normal_(self.u_emb.weight, 0.0, init_std)
        torch.nn.init.normal_(self.i_emb.weight, 0.0, init_std)
        self.biased = biased
        if biased:
            self.u_bias = torch.nn.Embedding(n_users, 1)
            self.i_bias = torch.nn.Embedding(n_items, 1)
            torch.nn.init.normal_(self.u_bias.weight, 0.0, init_std)
            torch.nn.init.normal_(self.i_bias.weight, 0.0, init_std)
        self.normalize = normalize

    def forward(self, u_idx, i_idx, dim=1):
        """
        It computes the scores for the given user-item pairs using the matrix factorization approach (dot product).
        :param u_idx: users for which the score has to be computed
        :param i_idx: items for which the score has to be computed
        :param dim: dimension along which the dot product has to be computed
        :return: predicted scores for given user-item pairs
        """
        pred = super().forward(u_idx, i_idx, dim)
        if self.normalize:
            pred = torch.sigmoid(pred)
        return pred


class DeepMatrixFactorization(torch.nn.Module):
    """
    Deep Matrix factorization model.
    The model is composed of two neural networks, one takes as input the user profile, and the other the item profile.
    Then, a user and an item embeddings are generated. The score for a user-item pair is computed by the cosine
    similarity between the two vectors.
    """

    def __init__(self, rating_matrix, n_layers, n_nodes_x_layer, act_func, n_factors, biased=False, init_std=0.001,
                 dropout_p=0., cosine=True):
        """
        Construction of the deep matrix factorization model.
        :param rating_matrix: user-item sparse matrix
        :param n_layers: number of layers in the deep architecture
        :param n_nodes_x_layer: number of nodes for each hidden layers
        :param act_func: activation function used on each hidden layer
        :param n_factors: size of embeddings for users and items
        :param biased: whether to use user and item biased
        :param init_std: standard deviation of the normal distribution used to initialize the model parameters
        :param dropout_p: percentage of units to be shut down at each layer
        :param cosine: whether to use cosine similarity or dot product to compute the final score
        """
        super(DeepMatrixFactorization, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.rating_matrix = rating_matrix
        common_sizes = [n_nodes_x_layer] * n_layers + [n_factors]
        user_sizes = [rating_matrix.shape[0]] + common_sizes
        item_sizes = [rating_matrix.shape[1]] + common_sizes
        self.user_layers = torch.nn.ModuleList([torch.nn.Linear(item_sizes[i - 1], item_sizes[i])
                                                for i in range(1, len(item_sizes))])
        self.item_layers = torch.nn.ModuleList([torch.nn.Linear(user_sizes[i - 1], user_sizes[i])
                                                for i in range(1, len(user_sizes))])
        self.act_func = act_func
        self.cosine = cosine
        self.cosine_sim = torch.nn.CosineSimilarity()
        self.dot_product = lambda u, i: torch.sum(u * i, dim=1, keepdim=True)
        self.biased = biased
        if biased and not cosine:
            self.user_bias = torch.nn.Embedding(rating_matrix.shape[0], 1)
            self.item_bias = torch.nn.Embedding(rating_matrix.shape[1], 1)
        self.initialize(init_std)

    def initialize(self, init_std):
        for layer in self.user_layers:
            torch.nn.init.normal_(layer.weight, 0.0, init_std)
            torch.nn.init.normal_(layer.bias, 0.0, init_std)
        for layer in self.item_layers:
            torch.nn.init.normal_(layer.weight, 0.0, init_std)
            torch.nn.init.normal_(layer.bias, 0.0, init_std)
        if self.biased and not self.cosine:
            torch.nn.init.normal_(self.user_bias.weight, 0.0, init_std)
            torch.nn.init.normal_(self.item_bias.weight, 0.0, init_std)

    def forward(self, u_idx, i_idx):
        """
        It computes the scores for the given user-item pairs using the deep matrix factorization model.
        :param u_idx: users for which the score has to be computed
        :param i_idx: items for which the score has to be computed
        :return: predicted scores for given user-item pairs
        """
        # get user-item rating vectors
        users = torch.tensor(self.rating_matrix[u_idx].toarray()).float()
        items = torch.tensor(self.rating_matrix[:, i_idx].transpose().toarray()).float()
        # compute user embedding
        for i, layer in enumerate(self.user_layers):
            self.dropout(users)
            users = layer(users)
            if i != len(self.user_layers) - 1:
                users = self.act_func(users)
        # compute item embedding
        for i, layer in enumerate(self.item_layers):
            self.dropout(items)
            items = layer(items)
            if i != len(self.item_layers) - 1:
                items = self.act_func(items)
        # compute user-item pair score
        if self.biased and not self.cosine:
            out = self.dot_product(users, items)
            out += self.user_bias(u_idx) + self.item_bias(i_idx)
            return out.squeeze()
        else:
            return self.cosine_sim(users, items)


class Trainer:
    """
    Abstract base class that any trainer must inherit from.
    """

    def __init__(self, model, optimizer, wandb_train=False):
        self.model = model
        self.optimizer = optimizer
        self.wandb_train = wandb_train

    def train(self, train_loader, val_loader, val_metric=None, n_epochs=200, early=None, verbose=10, save_path=None,
              model_name=None):
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
        concurrency is used to train simultaneously, so we need to distinguish among different concurrent runs
        """
        if val_metric is not None:
            check_metrics(val_metric)
        # if a regression metric is used, it has to be minimized
        best_val_score = sys.maxsize if "mse" in val_metric else 0.0
        early_counter = 0
        if self.wandb_train:
            # log gradients and parameters with Weights and Biases
            wandb.watch(self.model, log="all")

        for epoch in range(n_epochs):
            # training step
            train_loss, log_dict = self.train_epoch(train_loader)
            # validation step
            val_score = self.validate(val_loader, val_metric)
            # print epoch data
            if (epoch + 1) % verbose == 0:
                log_record = "Epoch %d - Train loss %.3f - Val %s %.3f" % (epoch + 1, train_loss, val_metric, val_score)
                if model_name is not None:
                    # add model name to log information if model name is available
                    log_record = ("%s: " % (model_name,)) + log_record
                # add log_dict to log_record
                log_record += " - " + " - ".join(["%s %.3f" % (k, v) for k, v in log_dict.items() if k != "train_loss"])
                # print epoch report
                print(log_record)
                if self.wandb_train:
                    # add to the log_dict returned from the training of the epoch (this information is different for
                    # every model) the information about the validation metric
                    wandb.log({"smooth_%s" % (val_metric,): val_score})
                    # log all stored information
                    wandb.log(log_dict)
            # save best model and update early stop counter, if necessary
            if ("mse" in val_metric and val_score < best_val_score) or \
                    ("mse" not in val_metric and val_score > best_val_score):
                best_val_score = val_score
                if self.wandb_train:
                    # the metric is logged only when a new best value is achieved for it
                    wandb.log({"%s" % (val_metric,): val_score})
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
        :return: training loss value averaged across training batches and a dictionary containing useful information
        to log, such as other metrics computed by this model
        """
        raise NotImplementedError()

    def predict(self, x, dim=1, **kwargs):
        """
        Method for performing a prediction of the model.
        :param x: tensor containing the user-item pair for which the prediction has to be computed. The first position
        is the user index, while the second position is the item index
        :param dim: dimension across which the dot product of the MF has to be computed
        :return: the prediction of the model for the given user-item pair
        """
        u_idx, i_idx = x[:, 0], x[:, 1]
        with torch.no_grad():
            return self.model(u_idx, i_idx, dim)

    def validate(self, val_loader, val_metric):
        """
        Method for validating the model.
        :param val_loader: data loader for validation data
        :param val_metric: validation metric name
        :return: validation score based on the given metric averaged across all validation examples
        """
        val_score, preds_vector, gt_vector = [], [], []
        if "@" in val_metric or val_metric == "auc":
            assert isinstance(val_loader, ValDataLoaderRanking), "You selected a ranking-based metric as validation" \
                                                                 "metric but your are not using the " \
                                                                 "ValDataLoaderRanking."
            for batch_idx, (X, y) in enumerate(val_loader):
                preds = self.predict(X.view(-1, 2))
                val_score.append(compute_metric(val_metric, preds.numpy().reshape(y.shape), y))
            val_score = np.mean(np.concatenate(val_score))
        else:
            assert isinstance(val_loader, ValDataLoaderRatings), "You selected a regression or classification metric" \
                                                                  " as validation metric but you are not using the " \
                                                                  "ValDataLoaderRatings."
            for batch_idx, (X, y) in enumerate(val_loader):
                if "fbeta" not in val_metric and val_metric != "acc":
                    preds = self.predict(X)
                    val_score.append(compute_metric(val_metric, preds.numpy(), y))
                else:
                    preds = self.predict(X, binarize=True)
                    preds_vector.append(preds)
                    gt_vector.append(y)
            if "fbeta" not in val_metric and val_metric != "acc":
                val_score = np.mean(np.concatenate(val_score))
                if val_metric == "rmse":
                    val_score = np.sqrt(val_score)
            else:
                preds_vector = np.concatenate(preds_vector)
                gt_vector = np.concatenate(gt_vector)
                val_score = compute_metric(val_metric, preds_vector, gt_vector)
                p, r, f, _ = precision_recall_fscore_support(gt_vector, preds_vector,
                                                             beta=float(val_metric.split("-")[1])
                                                             if "fbeta" in val_metric else 1.0, average=None)
                print("Neg prec %.3f - Pos prec %.3f - Neg rec %.3f - "
                      "Pos rec %.3f - Neg f %.3f - Pos f %.3f" % (p[0], p[1], r[0], r[1], f[0], f[1]))

                tn, fp, fn, tp = tuple(confusion_matrix(gt_vector, preds_vector).ravel())
                sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
                print("tn %d - fp %d - fn %d - tp %d" % (tn, fp, fn, tp))
                print("sensitivity %.3f - specificity %.3f" % (sensitivity, specificity))
                if self.wandb_train:
                    wandb.log({
                        "neg_prec": p[0],
                        "pos_prec": p[1],
                        "neg_rec": r[0],
                        "pos_rec": r[1],
                        "neg_f": f[0],
                        "pos_f": f[1],
                        "tn": tn,
                        "fp": fp,
                        "fn": fn,
                        "tp": tp,
                        "sensitivity": sensitivity,
                        "specificity": specificity
                    })

        return val_score

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
        pass
        # check_metrics(metrics)
        # if isinstance(metrics, str):
        #     metrics = [metrics]
        #
        # results = {m: [] for m in metrics}
        # if isinstance(test_loader, ValDataLoaderExact):
        #     with torch.no_grad():
        #         predicted_scores = torch.matmul(self.model.u_emb.weight.data, torch.t(self.model.i_emb.weight.data))
        #         if self.model.biased:
        #             predicted_scores = torch.add(predicted_scores, self.model.u_bias.weight.data)
        #             predicted_scores = torch.add(predicted_scores, torch.t(self.model.i_bias.weight.data))
        #     for batch_idx, (test_users, mask, ground_truth) in enumerate(test_loader):
        #         for m in results:
        #             results[m].append(
        #                 compute_metric(m, np.where(mask == 1, -np.inf, predicted_scores[test_users].numpy()),
        #                                ground_truth))
        #     # for batch_idx, (data, mask, ground_truth) in enumerate(test_loader):
        #     #     for m in results:
        #     # predicted_scores = self.predict(data.view(-1, 2))
        #     # results[m].append(compute_metric(m, predicted_scores.view(ground_truth.shape).numpy() * mask,
        #     #                                  ground_truth))
        # else:
        #     for batch_idx, (data, ground_truth) in enumerate(test_loader):
        #         for m in results:
        #             predicted_scores = self.predict(data.view(-1, 2))
        #             results[m].append(
        #                 compute_metric(m, predicted_scores.view(ground_truth.shape).numpy(), ground_truth))
        # for m in results:
        #     results[m] = np.mean(np.concatenate(results[m]))
        # return results


class MFTrainer(Trainer):
    def __init__(self, mf_model, optimizer, loss, wandb_train=False):
        """
        Constructor of the trainer for the MF model.
        :param mf_model: Matrix Factorization model
        :param optimizer: optimizer used for the training of the model
        :param wandb_train: whether the data has to be logged to WandB or not
        :param loss: loss that is used for traininng the Matrix Factorization model
        """
        super(MFTrainer, self).__init__(mf_model, optimizer, wandb_train)
        self.loss = loss

    def train_epoch(self, train_loader):
        # todo qui devo capire come usare la loss
        train_loss = 0.0
        for batch_idx, (u_i_pairs, ratings) in enumerate(train_loader):
            self.optimizer.zero_grad()
            loss = self.loss(self.model(u_i_pairs[:, 0], u_i_pairs[:, 1]), ratings)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        return train_loss / len(train_loader), {"train_loss": train_loss / len(train_loader)}


class MFTrainerBPR(Trainer):
    def __init__(self, mf_model, optimizer, wandb_train=False):
        """
        Constructor of the trainer for the MF model.
        :param mf_model: Matrix Factorization model
        :param optimizer: optimizer used for the training of the model
        :param wandb_train: whether the data has to be logged to WandB or not
        :param loss: loss that is used for traininng the Matrix Factorization model
        """
        super(MFTrainerBPR, self).__init__(mf_model, optimizer, wandb_train)
        self.act_func = torch.nn.LogSigmoid()
        self.loss = lambda pos, neg: -torch.sum(self.act_func(pos - neg))
        # self.loss = torch.nn.MarginRankingLoss(margin=2.0, reduction="sum")

    def train_epoch(self, train_loader):
        train_loss = 0.0
        for batch_idx, (u_i_pairs) in enumerate(train_loader):
            self.optimizer.zero_grad()
            feed_pairs = u_i_pairs.view(-1, 2)
            preds = self.model(feed_pairs[:, 0], feed_pairs[:, 1])
            preds = preds.view(u_i_pairs.shape[0], 2)
            loss = self.loss(preds[:, 0], preds[:, 1])
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        return train_loss / len(train_loader), {"train_loss": train_loss / len(train_loader)}


class MFTrainerRegression(MFTrainer):
    """
    Trainer for the Matrix Factorization model.

    This version of the trainer uses the MSE as loss function. It implements a regression task, where the objective
    is to minimize the distance between target and predicted ratings.
    """

    def __init__(self, mf_model, optimizer, loss=torch.nn.MSELoss(), wandb_train=False):
        """
        Constructor of the trainer for the MF model.
        :param mf_model: Matrix Factorization model
        :param optimizer: optimizer used for the training of the model
        :param wandb_train: whether the data has to be logged to WandB or not
        """
        super(MFTrainerRegression, self).__init__(mf_model, optimizer, loss, wandb_train)


class MFTrainerClassifier(MFTrainer):
    """
    Trainer for the Matrix Factorization model.

    This version of the trainer uses the focal loss as loss function. It implements a classification task, where the
    objective is to minimize the focal loss. The ratings are binary, hence they can be interpreted as binary classes.
    The objective is to discriminate between class 1 ("likes") and class 0/-1 ("dislikes").

    The focal loss is used since it is a generalization of binary cross entropy used to mitigate the class imbalance
    problem.
    """

    def __init__(self, mf_model, optimizer, loss=FocalLoss(alpha=0.5, gamma=0, reduction="mean"), wandb_train=False,
                 threshold=0.5):
        """
        Constructor of the trainer for the MF model.
        :param mf_model: Matrix Factorization model
        :param optimizer: optimizer used for the training of the model
        :param loss: loss function used to train the model
        :param wandb_train: whether the data has to be logged to WandB or not
        :param threshold: threshold used to determine whether an example is negative or positive
        """
        super(MFTrainerClassifier, self).__init__(mf_model, optimizer, loss, wandb_train)
        self.threshold = threshold

    def predict(self, x, dim=1, binarize=False):
        """
        Method for performing a prediction of the model.
        :param x: tensor containing the user-item pair for which the prediction has to be computed. The first position
        is the user index, while the second position is the item index
        :param dim: dimension across which the dot product of the MF has to be computed
        :param binarize: whether the prediction has to be binarized using the decision boundary. This is useful because
        some metrics requires the binarization, while other not
        :return: the prediction of the model for the given user-item pair
        """
        u_idx, i_idx = x[:, 0], x[:, 1]
        with torch.no_grad():
            # apply sigmoid because during training losses with logits are used for numerical stability
            pred = torch.sigmoid(self.model(u_idx, i_idx, dim))
            if binarize:
                pred = pred >= self.threshold
            return pred


class LTNTrainerMFRegression(Trainer):
    """
    Trainer for the Logic Tensor Network with Matrix Factorization as the predictive model for the Likes function.
    The Likes function takes as input a user-item pair and produce an un-normalized score (MF). Ideally, this score
    should be near 1 if the user likes the item, and near 0 if the user dislikes the item.
    The closeness between the predictions of the Likes function and the ground truth provided by the dataset for the
    training user-item pairs is obtained by maximizing the truth value of the predicate Sim. The predicate Sim takes
    as input a predicted score and the ground truth, and returns a value in the range [0., 1.]. Higher the value,
    higher the closeness between the predicted score and the ground truth.
    """

    def __init__(self, mf_model, optimizer, alpha, exp, p, wandb_train=False):
        """
        Constructor of the trainer for the LTN with MF as base model.
        :param mf_model: Matrix Factorization model to implement the Likes function
        :param optimizer: optimizer used for the training of the model
        :param alpha: coefficient for smoothing equality predicate
        :param gamma: exponent used in the equality predicate to give more weight to larger error, if needed
        :param p: hyper-parameter p of universal quantifier aggregator
        :param wandb_train: whether the information about the training of the model has to be logged on WandB or not
        """
        super(LTNTrainerMFRegression, self).__init__(mf_model, optimizer, wandb_train)
        self.Score = ltn.Function(self.model)
        self.Sim = ltn.Predicate(func=lambda pred, gt: torch.exp(-alpha * torch.pow(torch.abs(pred - gt), exp)))
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=p), quantifier='f')
        self.Forall_mean = ltn.Quantifier(ltn.fuzzy_ops.AggregMean(), quantifier='f')

    def train_epoch(self, train_loader):
        train_loss, train_sat_agg = 0.0, 0.0
        for batch_idx, (u, i, r) in enumerate(train_loader):
            self.optimizer.zero_grad()
            train_sat = self.Forall(ltn.diag(u, i, r), self.Sim(self.Score(u, i), r)).value
            train_sat_agg += train_sat.item()
            loss = 1. - train_sat
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_loader), {"training_overall_sat": train_sat_agg / len(train_loader)}


class LTNTrainerMFBPR(Trainer):
    """
    Trainer for the Logic Tensor Network with Matrix Factorization as the predictive model for the Likes function.
    The Likes function takes as input a user-item pair and produce an un-normalized score (MF). Ideally, this score
    should be near 1 if the user likes the item, and near 0 if the user dislikes the item.
    The closeness between the predictions of the Likes function and the ground truth provided by the dataset for the
    training user-item pairs is obtained by maximizing the truth value of the predicate Sim. The predicate Sim takes
    as input a predicted score and the ground truth, and returns a value in the range [0., 1.]. Higher the value,
    higher the closeness between the predicted score and the ground truth.
    """

    def __init__(self, mf_model, optimizer, alpha=10, p=2, wandb_train=False):
        """
        Constructor of the trainer for the LTN with MF as base model.
        :param mf_model: Matrix Factorization model to implement the Likes function
        :param optimizer: optimizer used for the training of the model
        :param alpha: coefficient for smoothing equality predicate
        :param gamma: exponent used in the equality predicate to give more weight to larger error, if needed
        :param p: hyper-parameter p of universal quantifier aggregator
        :param wandb_train: whether the information about the training of the model has to be logged on WandB or not
        """
        super(LTNTrainerMFBPR, self).__init__(mf_model, optimizer, wandb_train)
        self.Score = ltn.Function(self.model)
        # self.Dissim = ltn.Predicate(func=lambda pred_pos, pred_neg: torch.exp(
        #     -alpha * (1 / (pred_pos - pred_neg))))
        self.Sigmoid = torch.nn.Sigmoid()
        # todo provare sia con sigmoide e basta e anche con questo esponenziale
        # self.Dissim = ltn.Predicate(func=lambda pred_pos, pred_neg: torch.exp(-alpha * (1 / self.Sigmoid(pred_pos -
        #                                                                                                  pred_neg))))
        self.Dissim = ltn.Predicate(func=lambda pred_pos, pred_neg: self.Sigmoid(pred_pos - pred_neg))
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=p), quantifier='f')

    def train_epoch(self, train_loader):
        train_loss, train_sat_agg = 0.0, 0.0
        for batch_idx, (u, i_pos, i_neg) in enumerate(train_loader):
            self.optimizer.zero_grad()
            train_sat = self.Forall(ltn.diag(u, i_pos, i_neg), self.Dissim(self.Score(u, i_pos),
                                                                           self.Score(u, i_neg))).value
            train_sat_agg += train_sat.item()
            loss = 1. - train_sat
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_loader), {"training_overall_sat": train_sat_agg / len(train_loader)}


class LTNTrainerMFClassifier(Trainer):
    """
    Same trainer as the previous one, but for classification.
    """

    def __init__(self, mf_model, optimizer, p_pos=2, p_neg=2, p_sat_agg=2, wandb_train=False, threshold=0.5):
        """
        Constructor of the trainer for the LTN with MF as base model.
        :param mf_model: Matrix Factorization model to implement the Likes function
        :param optimizer: optimizer used for the training of the model
        :param p_pos: hyper-parameter p for universal quantifier aggregator used on positive examples
        :param p_neg: hyper-parameter p for universal quantifier aggregator used on negative examples
        :param wandb_train: whether the training information has to be logged on WandB or not
        :param threshold: threshold used for the decision boundary
        """
        super(LTNTrainerMFClassifier, self).__init__(mf_model, optimizer, wandb_train)
        self.Likes = ltn.Predicate(self.model)
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(), quantifier='f')
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.threshold = threshold
        self.sat_agg = ltn.fuzzy_ops.SatAgg(agg_op=ltn.fuzzy_ops.AggregPMeanError(p=p_sat_agg))

    def train_epoch(self, train_loader):
        train_loss, train_sat_agg, f1_sat, f2_sat = 0.0, 0.0, [], []
        for batch_idx, (u_pos, i_pos, u_neg, i_neg) in enumerate(train_loader):
            self.optimizer.zero_grad()
            if u_pos is not None and u_neg is not None:
                f1 = self.Forall(ltn.diag(u_pos, i_pos), self.Likes(u_pos, i_pos), p=self.p_pos)
                f2 = self.Forall(ltn.diag(u_neg, i_neg), self.Not(self.Likes(u_neg, i_neg)), p=self.p_neg)
                f1_sat.append(f1.value.item())
                f2_sat.append(f2.value.item())
                train_sat = self.sat_agg(f1, f2)
            elif u_pos is not None:
                f1 = self.Forall(ltn.diag(u_pos, i_pos), self.Likes(u_pos, i_pos), p=self.p_pos).value
                f1_sat.append(f1.item())
                train_sat = f1
            else:
                f2 = self.Forall(ltn.diag(u_neg, i_neg), self.Not(self.Likes(u_neg, i_neg)), p=self.p_neg).value
                f2_sat.append(f2.item())
                train_sat = f2
            train_sat_agg += train_sat.item()
            loss = 1. - train_sat
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_loader), {"training_overall_sat": train_sat_agg / len(train_loader),
                                                "pos_sat": np.mean(f1_sat),
                                                "neg_sat": np.mean(f2_sat)}

    def predict(self, x, dim=1, binarize=False):
        """
        Method for performing a prediction of the model.
        :param x: tensor containing the user-item pair for which the prediction has to be computed. The first position
        is the user index, while the second position is the item index
        :param dim: dimension across which the dot product of the MF has to be computed
        :param binarize: whether the prediction has to be binarized using the decision boundary. This is useful because
        some metrics requires the binarization, while other not
        :return: the prediction of the model for the given user-item pair
        """
        u_idx, i_idx = x[:, 0], x[:, 1]
        with torch.no_grad():
            # here, sigmoid does not have to be applied since LTN matrix factorization has the normalize parameter to
            # normalize the output if needed
            preds = self.model(u_idx, i_idx, dim)
            if binarize:
                preds = preds >= self.threshold
            return preds


class LTNTrainerMFClassifierTransferLearning(LTNTrainerMFClassifier):
    """
    Same trainer as the previous one, but that includes the formula for performing transfer learning.
    """

    def __init__(self, mf_model, optimizer, u_g_matrix, i_g_matrix, p_pos=2, p_neg=2, p_sat_agg=2, p_forall=2,
                 p_exists=2, wandb_train=False, threshold=0.5):
        """
        Constructor of the trainer for the LTN with MF as base model.
        :param mf_model: Matrix Factorization model to implement the Likes function
        :param optimizer: optimizer used for the training of the model
        :param u_g_matrix: users X genres matrix, containing the ratings predicted for each user-genre pair in the
        dataset. This is the pre-trained model.
        :param i_g_matrix: items X genres matrix, containing a 1 if the item i belongs to genre j, 0 otherwise
        :param p_pos: hyper-parameter p for universal quantifier aggregator used on positive examples
        :param p_neg: hyper-parameter p for universal quantifier aggregator used on negative examples
        :param p_forall: hyper-parameter p for universal quantifier aggregator used on second formula
        :param p_exists: hyper-parameter p for existential quantifier aggregator used on second formula
        :param wandb_train: whether the training information has to be logged on WandB or not
        :param threshold: threshold used for the decision boundary
        """
        super(LTNTrainerMFClassifierTransferLearning, self).__init__(mf_model, optimizer, p_pos, p_neg, p_sat_agg,
                                                                     wandb_train, threshold)
        self.And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        self.u_g_matrix = u_g_matrix
        self.LikesGenre = ltn.Predicate(func=lambda u_idx, g_idx: self.u_g_matrix[u_idx, g_idx])
        self.genres = ltn.Variable("genres", torch.tensor(range(self.u_g_matrix.shape[1])), add_batch_dim=False)
        self.i_g_matrix = i_g_matrix
        self.HasGenre = ltn.Predicate(func=lambda i_idx, g_idx: self.i_g_matrix[i_idx, g_idx])
        self.Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(), quantifier='e')
        self.p_forall = p_forall
        self.p_exists = p_exists

    def train_epoch(self, train_loader):
        train_loss, train_sat_agg, f1_sat, f2_sat, f3_sat = 0.0, 0.0, [], [], []
        for batch_idx, (u_pos, i_pos, u_neg, i_neg, users, items) in enumerate(train_loader):
            self.optimizer.zero_grad()
            # create LTN variables containing all batch interactions
            # users = ltn.Variable("users", torch.cat([u_pos.value, u_neg.value], dim=0), add_batch_dim=False)
            # items = ltn.Variable("items", torch.cat([i_pos.value, i_neg.value], dim=0), add_batch_dim=False)
            # compute SAT level of third formula
            f3 = self.Forall(ltn.diag(users, items),
                             self.Implies(
                                 self.Exists(self.genres,
                                             self.And(
                                                 self.Not(self.LikesGenre(users, self.genres)),
                                                 self.HasGenre(items, self.genres)
                                             ), p=self.p_exists),
                                 self.Not(self.Likes(users, items))),
                             p=self.p_forall)
            f3_sat.append(f3.value.item())
            if u_pos is not None and u_neg is not None:
                f1 = self.Forall(ltn.diag(u_pos, i_pos), self.Likes(u_pos, i_pos), p=self.p_pos)
                f2 = self.Forall(ltn.diag(u_neg, i_neg), self.Not(self.Likes(u_neg, i_neg)), p=self.p_neg)
                f1_sat.append(f1.value.item())
                f2_sat.append(f2.value.item())
                train_sat = self.sat_agg(f1, f2, f3)
            elif u_pos is not None:
                f1 = self.Forall(ltn.diag(u_pos, i_pos), self.Likes(u_pos, i_pos), p=self.p_pos).value
                f1_sat.append(f1.item())
                train_sat = self.sat_agg(f1, f3)
            else:
                f2 = self.Forall(ltn.diag(u_neg, i_neg), self.Not(self.Likes(u_neg, i_neg)), p=self.p_neg).value
                f2_sat.append(f2.item())
                train_sat = self.sat_agg(f2, f3)
            train_sat_agg += train_sat.item()
            loss = 1. - train_sat
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_loader), {"training_overall_sat": train_sat_agg / len(train_loader),
                                                "pos_sat": np.mean(f1_sat),
                                                "neg_sat": np.mean(f2_sat),
                                                "f3_sat": np.mean(f3_sat)}


class LTNTrainerMFGenres(MFTrainer):
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

    def __init__(self, mf_model, optimizer, alpha, p_exists, p_forall, n_movies, item_genres_matrix):
        """
        Constructor of the trainer for the LTN with MF as base model.
        :param mf_model: Matrix Factorization model to implement the Likes function
        :param optimizer: optimizer used for the training of the model
        :param alpha: coefficient of smooth equality predicate
        :param p_exists: hyper-parameter p for pMean of rule on genres
        :param p_forall: hyper-parameter p for pMeanError of rule on genres
        :param n_movies: number of movies in the dataset
        :param item_genres_matrix: sparse matrix with items on the rows and genres on the columns. A 1 means the item
        belongs to the genre
        """
        super(LTNTrainerMFGenres, self).__init__(mf_model, optimizer, alpha)
        self.p_forall = p_forall
        item_genres_matrix = torch.tensor(item_genres_matrix.todense())
        self.Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=p_exists), quantifier="e")
        self.And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        # here, we need to remove n_movies because the genres are in [n_movies, n_movies + n_genres] in the MF model
        # instead, in the item_genres_matrix they are in [0, n_genres]
        self.HasGenre = ltn.Predicate(func=lambda i_idx, g_idx: item_genres_matrix[i_idx, g_idx - n_movies])

    def train_epoch(self, train_loader):
        train_loss, train_sat_agg, f1_sat, f2_sat = 0.0, 0.0, 0.0, 0.0
        for batch_idx, ((u1, i1, r), (u2, i2, g, r_)) in enumerate(train_loader):
            self.optimizer.zero_grad()
            f1 = self.Forall(ltn.diag(u1, i1, r), self.Sim(self.Likes(u1, i1), r)).value
            f2 = self.Forall(ltn.diag(u2, i2), self.Implies(
                self.Exists(g, self.And(self.Sim(self.Likes(u2, g), r_), self.HasGenre(i2, g))),
                self.Sim(self.Likes(u2, i2), r_)), p=self.p_forall).value
            train_sat = self.sat_agg(f1, f2)
            train_sat_agg += train_sat.item()
            loss = 1. - train_sat
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            f1_sat += f1.item()
            f2_sat += f2.item()

        return train_loss / len(train_loader), {"training_overall_sat": train_sat_agg / len(train_loader),
                                                "training_sat_axiom_1": f1_sat / len(train_loader),
                                                "training_sat_axiom_2": f2_sat / len(train_loader)}


class LTNTrainerMFGenresNew(MFTrainer):
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

    This model includes also an additional logical function, called LikesGenre. This is a pre-trained function which
    returns values near 1 if the input user-genre pair is compatible, 0 otherwise. This function is used to instruct
    the model on recommending based on content information when no user-item interactions are available (cold-start
    cases or very sparse cases). The idea is that `n` user-item pairs are randomly sampled from the user-item matrix.
    Since the matrix is really sparse, the probability to sample missing ratings is really high. On that ratings we
    apply the second axiom. If, by chance, we sample existing ratings (the probability is low but it is not impossible),
    the rule will act as a kind of regularization when the ground truth is not aligned with the LikesGenre function.

    For example, if the LikesGenre function states that a user does not like a genre, while the ground truth states that
    the user likes a movie with that genre, the LikesGenre will correct the ground truth. It is possible to add a weight
    on the loss to prevent the LikesGenre to dramatically change the ground truth.
    """

    def __init__(self, mf_model, optimizer, alpha, p_exists, p_forall, item_genres_matrix, likes_genre_model,
                 likes_genre_path):
        """
        Constructor of the trainer for the LTN MF GENRES model.

        :param mf_model: Matrix Factorization model to implement the Likes function
        :param optimizer: optimizer used for the training of the model
        :param alpha: coefficient of smooth equality predicate
        :param p_exists: hyper-parameter p for pMean of rule on genres
        :param p_forall: hyper-parameter p for pMeanError of rule on genres
        :param item_genres_matrix: sparse matrix with items on the rows and genres on the columns. A 1 means the item
        belongs to the genre
        :param likes_genre_model: model to be used for the LikesGenre function. This model is not learnable. It is a
        pre-trained model.
        :param likes_genre_path: path where the weights for the pre-trained LikesGenre model are stored.
        """
        # todo ricordarsi di freezare il predicato
        # todo costruire la item_genres_matrix
        super(LTNTrainerMFGenresNew, self).__init__(mf_model, optimizer, alpha)
        self.p_forall = p_forall
        item_genres_matrix = torch.tensor(item_genres_matrix.todense())
        self.Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=p_exists), quantifier="e")
        self.And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        self.HasGenre = ltn.Predicate(func=lambda i_idx, g_idx: item_genres_matrix[i_idx, g_idx])
        likes_genre_model.load_state_dict(torch.load(likes_genre_path)["model_state_dict"])
        # freeze LikesGenre function weights since it is a pre-trained function
        likes_genre_model.requires_grad_(False)
        self.LikesGenre = ltn.Function(likes_genre_model)

    def train_epoch(self, train_loader):
        train_loss, train_sat_agg, f1_sat, f2_sat = 0.0, 0.0, 0.0, 0.0
        for batch_idx, ((u1, i1, r), (u2, i2, g, r_)) in enumerate(train_loader):
            self.optimizer.zero_grad()
            f1 = self.Forall(ltn.diag(u1, i1, r), self.Sim(self.Likes(u1, i1), r)).value
            # todo qua c'e' il discorso di cambiare il p per concentrarsi moltissimo sugli outliers - perche' abbiamo veramente pochi esempi in cui la regola vale
            # todo verificare perche' la regola non funziona. Sembrerebbe che le coppie utente item pescate a caso siano gia' negative, nel senso che, in media, all'utente non piace l'item
            # todo sembrerebbe anche che, in media, agli utenti del batch non piacciano i generi che sono stati pescati
            # todo HasGenre e' spesso falso e ci sta, il discorso e' che e' il per ogni che e' molto basso
            # todo applicare solo da una certa epoca in poi -> questo potrebbe essere interessante
            # todo filtrare i generi puo' essere un'idea, cosi diventa meno rara la soddisfazione automatica della regola e anche piu' accurato il predicato LikesGenre, che pio' essere addestrato con ndcg@1 se ci sono solo 18 generi
            # todo pensare a tipologie di implicazione che non sono sempre vere a caso -> provare a dare un'occhiata al paper di Emilie
            # todo per il sampling dovrei aumentare la probabilita' che all'utente piace un film, invece sembra che non sia cosi su quelli che pesco
            # todo pensiamo bene alle varie parti della regola e a cosa puo' esserci che non va, possiamo fare affidamento su predicato appreso dei generi? E' veramente accurato?
            # todo quali dovrebbero essere i valori di p? p puo' esserci utile per rendere meno vera la regola? Abbiamo sempre il problema che la premessa e' falsa. In piu' e' raro pescare qualcosa che all'utente piace e quindi la regola non si puo' applicare
            # todo forse possiamo applicare anche la regola che se gli piace un genere allora gli piacciono i film di quel genere, questa meno strict dell'altra
            # todo possiamo anche pensare di iniziare ad utilizzare queste regole piu' tardi durante il learning
            # todo mi basta riuscire a mostrare che questa conoscenza e' utile e che LTN e' una maniera semplice di utilizzarla per fare in modo che ci possa essere un paper
            # todo sarebbe figo anche far vedere con LTN che siccome la regola e' gia' soddisfatta dai dati, allora la MF e' in grado di apprendere l'informazione sui generi
            # todo iniziare ad utilizzare la regola da un certo punto in poi o usare annealing
            # esistono dei tipi di implicazione attenti a queste cose o si possono creare?
            f2 = self.Forall(ltn.diag(u2, i2), self.Implies(
                self.Exists(g, self.And(self.Sim(self.LikesGenre(u2, g), r_), self.HasGenre(i2, g))),
                self.Sim(self.Likes(u2, i2), r_)), p=self.p_forall).value
            print("self.Forall(ltn.diag(u2, i2), self.Sim(self.Likes(u2, i2), r_))")
            print(self.Forall(ltn.diag(u2, i2), self.Sim(self.Likes(u2, i2), r_)).value)
            print("self.Exists(g, self.Sim(self.LikesGenre(u2, g), r_))")
            print(self.Exists(g, self.Sim(self.LikesGenre(u2, g), r_)).value)
            print("self.Exists(g, self.HasGenre(i2, g))")
            print(self.Exists(g, self.HasGenre(i2, g)).value)
            print("self.Exists(g, self.And(self.Sim(self.LikesGenre(u2, g), r_), self.HasGenre(i2, g)))")
            print(self.Exists(g, self.And(self.Sim(self.LikesGenre(u2, g), r_), self.HasGenre(i2, g))).value)
            print("self.Implies(self.Exists(g, self.And(self.Sim(self.LikesGenre(u2, g), r_), self.HasGenre(i2, g))),self.Sim(self.Likes(u2, i2), r_))")
            print(self.Implies(
                self.Exists(g, self.And(self.Sim(self.LikesGenre(u2, g), r_), self.HasGenre(i2, g))),
                self.Sim(self.Likes(u2, i2), r_)).value)
            train_sat = f1  # self.sat_agg(f1, f2)
            train_sat_agg += train_sat.item()
            loss = 1. - train_sat
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            f1_sat += f1.item()
            f2_sat += f2.item()

        return train_loss / len(train_loader), {"training_overall_sat": train_sat_agg / len(train_loader),
                                                "training_sat_axiom_1": f1_sat / len(train_loader),
                                                "training_sat_axiom_2": f2_sat / len(train_loader)}


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

    def grid_search(self, dataset_id, local_dataset_path_prefix, local_config_path_prefix, wandb_project):
        """
        It performs a grid search of the model using a Weights and Biases sweep.
        The dataset_id is used to set the seed for the experiment and also for downloading the dataset artifact from
        wandb servers.
        The grid search is performed using random search with early stopping. The configuration file for each model has
        to be stored in ./config. The configuration file specifies the hyper-parameters that have to be tested in the
        sweep (they are stored under the key `sweep_config` which specifies also the sweep strategy (random search))
        as well as the parameters for the training, evaluation and test of the model. The random search is performed for
        the number of run specified in the configuration file under the key `search_n_iter`.

        :param dataset_id: name of the dataset artifact in wandb (mode-dataset_name-n_neg-seed)
        :param local_dataset_path_prefix: local dataset path prefix. It is used to download the dataset artifact there
        :param local_config_path_prefix: local configuration path prefix. It is where the best configuration JSON files
        are saved
        :param wandb_project: name of the wandb project
        :return:
        """
        # check if a best configuration file for this model and dataset already exists
        if os.path.exists("%s/config-model=%s-dataset=%s.json" % (local_config_path_prefix,
                                                                  self.model_name, dataset_id)):
            print("Grid search for model=%s-dataset=%s has been already performed. Find the config "
                  "file at %s/config-model=%s-dataset=%s.json" % (self.model_name, dataset_id,
                                                                  local_config_path_prefix, self.model_name,
                                                                  dataset_id))
        else:
            # if the best configuration artifact does not exists, we have to run a grid search for this model and
            # dataset
            # create the sweep for this grid search
            # take sweep config
            sweep_config = self.param_config["sweep_config"]
            # add sweep name to the config
            sweep_config["name"] = "grid_search:model=%s-dataset=%s" % (self.model_name, dataset_id)
            # create the sweep object and get the sweep identifier
            sweep_id = wandb.sweep(sweep=sweep_config, project=wandb_project)

            def single_run():
                """
                This function defines a single grid search run with randomly sampled hyper-parameter values.
                """
                # create a wandb run for performing this single grid search run
                with wandb.init(project=wandb_project, job_type="grid_search") as run:
                    # load the dataset from disk
                    with open("%s/%s" % (local_dataset_path_prefix, dataset_id,), 'rb') as dataset_file:
                        dataset = pickle.load(dataset_file)
                    # run the train of the model
                    self.grid_search_train(run, dataset_id, dataset)

            # set seed for reproducible experiments
            # todo qua non ha senso il seed, tanto non funziona sugli sweep, pero' almeno so con che seed riprodurre
            #  la specifica run
            set_seed(int(dataset_id.split("_")[-1]))  # the seed is the last information in every dataset name
            # run the wandb agent that will perform the sweep
            wandb.agent(sweep_id, function=single_run, count=self.param_config["search_n_iter"])

            best_config = api.sweep("%s/%s" % (wandb_project, sweep_id)).best_run().config

            print("Saving best hyper-parameters "
                  "configuration file at %s/config-model=%s-dataset=%s.json" % (local_config_path_prefix,
                                                                                self.model_name,
                                                                                dataset_id))
            with open("%s/config-model=%s-dataset=%s.json" % (local_config_path_prefix, self.model_name,
                                                              dataset_id), "w") as outfile:
                json.dump(best_config, outfile, indent=4)

    def grid_search_train(self, wandb_run, dataset_id, data):
        """
        It implements the training procedure of the grid search done using a wandb sweep.

        This method will differ among models.

        :param wandb_run: wandb run that has to run the train
        :param dataset_id: name of the dataset, used to log
        :param data: dataset on which the grid search run has to be computed
        :return:
        """
        pass

    def train(self, dataset_id, config_seed, local_dataset_path_prefix, local_config_path_prefix,
              local_model_path_prefix, wandb_project):
        """
        It performs the training of the model with the best configuration of hyper-parameters found using a grid search.
        If a best configuration wandb artifact does not exist for this model already, the method automatically calls the
        grid search procedure for this model in such a way to create a best configuration file.

        The dataset_id is used to set the seed for the training and also for downloading the dataset artifact from
        wandb servers.

        The training is performed with the hyper-parameters specified in the best configuration file and with the
        parameters specified in the corresponding configuration file contained in ./config.

        The training procedure creates a best model artifact on Weights and Biases for future use. Also, it creates a
        result artifact, containing a JSON file that reports the test metrics connected to this training procedure.

        :param dataset_id: name of the dataset artifact in wandb (mode-dataset_name-n_neg-seed)
        :param config_seed: seed of the best configuration artifact that has to be used to perform the training of the
        model.
        If the file with the given seed for the given dataset exists, the training will be performed. If it does not
        exist, the method will raise an Exception.
        :param local_dataset_path_prefix: local dataset path prefix. It is used to download the dataset artifact there
        :param local_config_path_prefix: local configuration path prefix. It is where the best configuration JSON files
        are stored
        :param local_model_path_prefix: local model path prefix. It is where the best model file is stored
        :param wandb_project: name of the wandb project
        :return:
        """
        # check if a best model file for this model already exists, it is not necessary to train it another time
        if os.path.exists(os.path.join(local_model_path_prefix, "model-model=%s-dataset=%s.pth" % (self.model_name,
                                                                                                   dataset_id))):
            print("Best model training for model=%s-dataset=%s has been already performed. Find the model file"
                  "at %s/model-model=%s-dataset=%s.pth" % (self.model_name, dataset_id, local_model_path_prefix,
                                                           self.model_name, dataset_id))
        else:
            # if the best model file does not exist, we have to perform the training for this model
            # check if a best configuration file exists
            if not os.path.exists(os.path.join(local_config_path_prefix, "config-model=%s-dataset=%s.json" %
                                                                         (self.model_name,
                                                                          remove_seed_from_dataset_name(dataset_id)
                                                                          + str(config_seed)))):
                # if the best configuration file does not exists with the given seed, we have to raise an exception
                raise FileNotFoundError("The configuration file with name config-model=%s-dataset=%s.json does "
                                        "not exist. Please, run a grid search to generate the requested "
                                        "file." % (self.model_name, remove_seed_from_dataset_name(dataset_id)
                                                   + str(config_seed)))
            else:
                # run the training of the model
                # set seed from reproducibility
                set_seed(int(dataset_id.split("_")[-1]))  # the seed is the last character in the string
                # if the best configuration file already exists, we can train the model with the hyper-parameters
                # specified in the configuration
                with wandb.init(project=wandb_project, job_type="model_training") as run:
                    run.name = "model_training:model=%s-dataset=%s" % (self.model_name, dataset_id)
                    # load hyper-parameters values from configuration file
                    with open("%s/config-model=%s-dataset=%s.json" % (local_config_path_prefix, self.model_name,
                                                                      remove_seed_from_dataset_name(dataset_id)
                                                                      + str(config_seed))) as json_file:
                        best_config = json.load(json_file)
                    # load dataset
                    with open("%s/%s" % (local_dataset_path_prefix, dataset_id), 'rb') as dataset_file:
                        dataset = pickle.load(dataset_file)
                    # train the model on the downloaded dataset with the downloaded best configuration
                    # this method will also create a best model file and save it in the local_model_path_prefix
                    self.train_model(dataset, dataset_id, best_config, local_model_path_prefix)

    def train_model(self, data, dataset_id, best_config, local_model_path_prefix):
        """
        It performs the training of the model on the given dataset with the given configuration of hyper-parameters.
        It saved the best model on the given path and also creates an artifact for the best model on Weights and Biases.

        :param data: dataset on which the training has to be performed
        :param dataset_id: name of dataset used to log information while training
        :param best_config: best configuration to load the hyper-parameters values
        :param local_model_path_prefix: local model path where to save the best model
        :return:
        """
        pass

    def test(self, dataset_id, config_seed, local_dataset_path_prefix, local_config_path_prefix,
             local_model_path_prefix, local_result_path_prefix, wandb_project):
        """
        It tests the performances of a trained model on the test set. After the test of the model has been done, it
        creates a wandb artifact that contains a JSON file reporting the test metrics.

        Before testing the model, it loads the model from a best model artifact. If a best model artifact does not
        exist for this model, it raises an exception.

        Parameter dataset_id is used to download the corresponding dataset artifact in such a way to perform the test
        of the model.

        :param dataset_id: name of the dataset artifact in wandb (mode-dataset_name-n_neg-seed)
        :param config_seed: seed of the best configuration artifact that has to be used to perform the test of the
        model. The model is created using the parameters in the configuration and then the best model weights are
        loaded.
        :param local_dataset_path_prefix: local dataset path prefix. It is used to download the dataset artifact there
        :param local_config_path_prefix: local config path prefic. It is used to download the best configuration file
        there. The configuration serves to create the model and load the weights.
        :param local_model_path_prefix: local model path prefix. It is where the best model file is stored
        :param local_result_path_prefix: local result path prefix. It is where the result file is stored
        :param wandb_project: name of the wandb project
        :return:
        """
        # check if a result file for this model already exists
        if os.path.exists(os.path.join(local_result_path_prefix, "result-model=%s-dataset=%s.json" % (self.model_name,
                                                                                                      dataset_id))):
            print("Result file result-model=%s-dataset=%s.json already exists." % (self.model_name, dataset_id))
        else:
            # if it does not exists, we have to create it
            # check if a best model artifact for this model already exists
            if not os.path.exists(
                    os.path.join(local_model_path_prefix, "model-model=%s-dataset=%s.pth" % (self.model_name,
                                                                                             dataset_id))):
                raise ValueError("Best model file model-model=%s-dataset=%s.pth does not exist. You should train"
                                 "the model before testing it." % (self.model_name, dataset_id))
            else:
                # if the best model file exists, we can run the test of the model
                # load the dataset from disk
                with open("%s/%s" % (local_dataset_path_prefix, dataset_id), 'rb') as dataset_file:
                    dataset = pickle.load(dataset_file)
                # load hyper-parameters values from configuration file
                with open("%s/config-model=%s-dataset=%s.json" % (local_config_path_prefix, self.model_name,
                                                                  remove_seed_from_dataset_name(dataset_id)
                                                                  + str(config_seed))) as json_file:
                    best_config = json.load(json_file)
                # test the model - it also loads the best model file on the torch model and creates the result
                # file
                print("Testing model=%s-dataset=%s" % (self.model_name, dataset_id))
                self.test_model(dataset, dataset_id, best_config, local_model_path_prefix, local_result_path_prefix)

    def test_model(self, data, dataset_id, best_config, local_model_path_prefix, local_result_path_prefix):
        """
        It performs the test of the model on the given dataset and saves the results at the given path.
        It constructs the model with the best configuration and loads the best weights.

        :param data: dataset on which the performances have to be tested
        :param dataset_id: name of dataset used to load the best model file
        :param best_config: best configuration dictionary, used to create the model to then loads the best weights
        :param local_model_path_prefix: local model path prefix. It is used to find the best model file on disk
        :param local_result_path_prefix: local result path prefix. It is the path where the result file of this test has
        to be saved
        :return:
        """
        pass


class StandardMFModel(Model):
    """
    Standard matrix factorization model. It uses the MSE between target and predicted ratings as an objective.
    """

    def __init__(self, config_file_name="standard_mf"):
        super(StandardMFModel, self).__init__(config_file_name)

    def grid_search_train(self, wandb_run, dataset_id, data):
        # get a random hyper-parameter configuration to be tested
        k = wandb.config.k
        biased = wandb.config.biased
        lr = wandb.config.lr
        tr_batch_size = wandb.config.tr_batch_size
        wd = wandb.config.wd

        # change run name for log purposes
        wandb_run.name = "grid_search:model=%s-dataset=%s-config:k=%d,biased=%d,lr=%.4f,tr_batch_size=%d,wd=%.4f" % \
                         (self.model_name, dataset_id, k, biased, lr, tr_batch_size, wd)

        # train model with current configuration
        model = MatrixFactorization(data.n_users, data.n_items, k, biased)
        tr_loader = TrainingDataLoader(data.train, tr_batch_size)
        val_loader = ValDataLoader(data.val, self.param_config["val_batch_size"])
        trainer = MFTrainer(model, Adam(model.parameters(), lr=lr, weight_decay=wd))
        trainer.train(tr_loader, val_loader, self.param_config["val_metric"],
                      n_epochs=self.param_config["n_epochs"],
                      verbose=self.param_config["verbose"],
                      wandb_train=True, model_name=wandb_run.name, early=self.param_config["early_stop"])

    def train_model(self, data, dataset_id, best_config, local_model_path_prefix):
        # train model with best configuration
        model = MatrixFactorization(data.n_users, data.n_items, best_config["k"], best_config["biased"])
        tr_loader = TrainingDataLoader(data.train, best_config["tr_batch_size"])
        val_loader = ValDataLoader(data.val, self.param_config["val_batch_size"])
        trainer = MFTrainer(model, Adam(model.parameters(), lr=best_config["lr"], weight_decay=best_config["wd"]))
        trainer.train(tr_loader, val_loader, self.param_config["val_metric"],
                      n_epochs=self.param_config["n_epochs"],
                      verbose=self.param_config["verbose"],
                      wandb_train=True, model_name="model_training:model=%s-dataset=%s" % (self.model_name, dataset_id),
                      early=self.param_config["early_stop"],
                      save_path="%s/model-model=%s-dataset=%s.pth" % (local_model_path_prefix, self.model_name,
                                                                      dataset_id))

    def test_model(self, data, dataset_id, best_config, local_dataset_path_prefix, local_result_path_prefix):
        # create test loader and model
        test_loader = ValDataLoader(data.test, self.param_config["val_batch_size"])
        model = MatrixFactorization(data.n_users, data.n_items, best_config["k"], best_config["biased"])
        trainer = MFTrainer(model, Adam(model.parameters(), lr=best_config["lr"], weight_decay=best_config["wd"]))
        # load best weights on the model
        trainer.load_model("%s/model-model=%s-dataset=%s.pth" % (local_dataset_path_prefix, self.model_name,
                                                                 dataset_id))
        # test the model
        metrics_dict = trainer.test(test_loader, self.param_config["test_metrics"])
        # create the result JSON file and save it on disk
        with open("%s/result-model=%s-dataset=%s.json" % (local_result_path_prefix,
                                                          self.model_name, dataset_id), "w") as outfile:
            json.dump(metrics_dict, outfile, indent=4)


class LTNMFModel(Model):
    """
    Matrix factorization model trained using LTN. It uses a logic formula that forces the target ratings to be as
    similar as possible to the predicted ratings as an objective.
    """

    def __init__(self, config_file_name="ltn_mf"):
        super(LTNMFModel, self).__init__(config_file_name)

    def grid_search_train(self, wandb_run, dataset_id, data):
        # get a random hyper-parameter configuration to be tested
        k = wandb.config.k
        biased = wandb.config.biased
        lr = wandb.config.lr
        tr_batch_size = wandb.config.tr_batch_size
        wd = wandb.config.wd
        p = wandb.config.p
        alpha = wandb.config.alpha

        # change run name for log purposes
        wandb_run.name = "grid_search:model=%s-dataset=%s-config:k=%d,biased=%d,lr=%.4f,tr_batch_size=%d," \
                         "wd=%.4f,p=%d,alpha=%.3f" % (
                             self.model_name, dataset_id, k, biased, lr, tr_batch_size, wd, p, alpha)

        # train model with current configuration
        model = MatrixFactorization(data.n_users, data.n_items, k, biased)
        tr_loader = TrainingDataLoaderLTN(data.train, tr_batch_size)
        val_loader = ValDataLoader(data.val, self.param_config["val_batch_size"])
        trainer = LTNTrainerMF(model, Adam(model.parameters(), lr=lr, weight_decay=wd),
                               alpha=alpha, p=p)
        trainer.train(tr_loader, val_loader, self.param_config["val_metric"],
                      n_epochs=self.param_config["n_epochs"],
                      verbose=self.param_config["verbose"],
                      wandb_train=True, model_name=wandb_run.name, early=self.param_config["early_stop"])

    def train_model(self, data, dataset_id, best_config, local_model_path_prefix):
        # train model with best configuration
        model = MatrixFactorization(data.n_users, data.n_items, best_config["k"], best_config["biased"])
        tr_loader = TrainingDataLoaderLTN(data.train, best_config["tr_batch_size"])
        val_loader = ValDataLoader(data.val, self.param_config["val_batch_size"])
        trainer = LTNTrainerMF(model, Adam(model.parameters(), lr=best_config["lr"], weight_decay=best_config["wd"]),
                               alpha=best_config["alpha"], p=best_config["p"])
        trainer.train(tr_loader, val_loader, self.param_config["val_metric"],
                      n_epochs=self.param_config["n_epochs"],
                      verbose=self.param_config["verbose"],
                      wandb_train=True, model_name="model_training:model=%s-dataset=%s" % (self.model_name, dataset_id),
                      early=self.param_config["early_stop"],
                      save_path="%s/model-model=%s-dataset=%s.pth" % (local_model_path_prefix, self.model_name,
                                                                      dataset_id))

    def test_model(self, data, dataset_id, best_config, local_dataset_path_prefix, local_result_path_prefix):
        # create test loader and model
        test_loader = ValDataLoader(data.test, self.param_config["val_batch_size"])
        model = MatrixFactorization(data.n_users, data.n_items, best_config["k"], best_config["biased"])
        trainer = LTNTrainerMF(model, Adam(model.parameters(), lr=best_config["lr"], weight_decay=best_config["wd"]),
                               alpha=best_config["alpha"], p=best_config["p"])
        # load best weights on the model
        trainer.load_model("%s/model-model=%s-dataset=%s.pth" % (local_dataset_path_prefix, self.model_name,
                                                                 dataset_id))
        # test the model
        metrics_dict = trainer.test(test_loader, self.param_config["test_metrics"])
        # create the result JSON file and save it on disk
        with open("%s/result-model=%s-dataset=%s.json" % (local_result_path_prefix,
                                                          self.model_name, dataset_id), "w") as outfile:
            json.dump(metrics_dict, outfile, indent=4)


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

    def grid_search_train(self, wandb_run, dataset_id, data):
        # get a random hyper-parameter configuration to be tested
        k = wandb.config.k
        biased = wandb.config.biased
        lr = wandb.config.lr
        tr_batch_size = wandb.config.tr_batch_size
        wd = wandb.config.wd
        p = wandb.config.p
        n_sampled_genres = wandb.config.n_sampled_genres
        exists = wandb.config.exists
        alpha = wandb.config.alpha

        # change run name for log purposes
        wandb_run.name = "grid_search:model=%s-dataset=%s-config:k=%d,biased=%d,lr=%.4f,tr_batch_size=%d," \
                         "wd=%.4f,p=%d,n_sampled_genres=%d,exists=%d,alpha=%.3f" % (
                             self.model_name, dataset_id, k, biased, lr,
                             tr_batch_size, wd, p, n_sampled_genres, exists, alpha)

        # train model with current configuration
        model = MatrixFactorization(data.n_users, data.n_items, k, biased)
        tr_loader = TrainingDataLoaderLTNGenres(data.train, data.n_users, data.n_items - data.n_genres, data.n_genres,
                                                n_sampled_genres, tr_batch_size)
        val_loader = ValDataLoader(data.val, self.param_config["val_batch_size"])
        trainer = LTNTrainerMFGenres(model, Adam(model.parameters(), lr=lr, weight_decay=wd),
                                     alpha=alpha, p=p, n_movies=data.n_items - data.n_genres,
                                     item_genres_matrix=data.item_genres_matrix, exists=exists)
        trainer.train(tr_loader, val_loader, self.param_config["val_metric"],
                      n_epochs=self.param_config["n_epochs"],
                      verbose=self.param_config["verbose"],
                      wandb_train=True, model_name=wandb_run.name, early=self.param_config["early_stop"])

    def train_model(self, data, dataset_id, best_config, local_model_path_prefix):
        # train model with best configuration
        model = MatrixFactorization(data.n_users, data.n_items, best_config["k"], best_config["biased"])
        tr_loader = TrainingDataLoaderLTNGenres(data.train, data.n_users, data.n_items - data.n_genres, data.n_genres,
                                                best_config["n_sampled_genres"], best_config["tr_batch_size"])
        val_loader = ValDataLoader(data.val, self.param_config["val_batch_size"])
        trainer = LTNTrainerMFGenres(model, Adam(model.parameters(), lr=best_config["lr"],
                                                 weight_decay=best_config["wd"]),
                                     alpha=best_config["alpha"], p=best_config["p"],
                                     n_movies=data.n_items - data.n_genres,
                                     item_genres_matrix=data.item_genres_matrix, exists=best_config["exists"])
        trainer.train(tr_loader, val_loader, self.param_config["val_metric"],
                      n_epochs=self.param_config["n_epochs"],
                      verbose=self.param_config["verbose"],
                      wandb_train=True, model_name="model_training:model=%s-dataset=%s" % (self.model_name, dataset_id),
                      early=self.param_config["early_stop"],
                      save_path="%s/model-model=%s-dataset=%s.pth" % (local_model_path_prefix, self.model_name,
                                                                      dataset_id))

    def test_model(self, data, dataset_id, best_config, local_dataset_path_prefix, local_result_path_prefix):
        # create test loader and model
        test_loader = ValDataLoader(data.test, self.param_config["val_batch_size"])
        model = MatrixFactorization(data.n_users, data.n_items, best_config["k"], best_config["biased"])
        trainer = LTNTrainerMFGenres(model, Adam(model.parameters(), lr=best_config["lr"],
                                                 weight_decay=best_config["wd"]),
                                     alpha=best_config["alpha"], p=best_config["p"],
                                     n_movies=data.n_items - data.n_genres,
                                     item_genres_matrix=data.item_genres_matrix, exists=best_config["exists"])
        # load best weights on the model
        trainer.load_model("%s/model-model=%s-dataset=%s.pth" % (local_dataset_path_prefix, self.model_name,
                                                                 dataset_id))
        # test the model
        metrics_dict = trainer.test(test_loader, self.param_config["test_metrics"])
        # create the result JSON file and save it on disk
        with open("%s/result-model=%s-dataset=%s.json" % (local_result_path_prefix,
                                                          self.model_name, dataset_id), "w") as outfile:
            json.dump(metrics_dict, outfile, indent=4)
