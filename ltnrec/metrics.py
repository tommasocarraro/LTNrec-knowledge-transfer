import numpy as np
from sklearn.metrics import fbeta_score, accuracy_score
valid_metrics = ['ndcg', 'recall', 'hit', 'auc', 'mse', 'rmse', 'fbeta', 'acc']
# todo da notare che precision e recall non hanno alcun senso in caso di leave-one-out evaluation perche' il target
#  item e' sempre uno, quindi, se l'unico target item e' nei primi k, la precision vale 1/k e la recall vale 1


def hit_at_k(pred_scores, ground_truth, k=10):
    """
    Computes the hit ratio (at k) given the predicted scores and relevance of the items.
    :param pred_scores: score vector in output from the recommender (unsorted ranking)
    :param ground_truth: binary vector with relevance data (1 relevant, 0 not relevant)
    :param k: length of the ranking on which the metric has to be computed
    :return: hit ratio at k position
    """
    k = min(pred_scores.shape[1], k)
    # generate ranking
    rank = np.argsort(-pred_scores, axis=1)
    # get relevance of first k items in the ranking
    rank_relevance = ground_truth[np.arange(pred_scores.shape[0])[:, np.newaxis], rank[:, :k]]
    # sum along axis 1 to count number of relevant items on first k-th positions
    # it is enough to have one relevant item in the first k-th for having a hit ratio of 1
    return rank_relevance.sum(axis=1) > 0


def ndcg_at_k(pred_scores, ground_truth, k=10):
    """
    Computes the NDCG (at k) given the predicted scores and relevance of the items.
    :param pred_scores: score vector in output from the recommender (unsorted ranking)
    :param ground_truth: binary vector with relevance data (1 relevant, 0 not relevant)
    :param k: length of the ranking on which the metric has to be computed
    :return: NDCG at k position
    """
    k = min(pred_scores.shape[1], k)
    # compute DCG
    # generate ranking
    rank = np.argsort(-pred_scores, axis=1)
    # get relevance of first k items in the ranking
    rank_relevance = ground_truth[np.arange(pred_scores.shape[0])[:, np.newaxis], rank[:, :k]]
    log_term = 1. / np.log2(np.arange(2, k + 2))
    # compute metric
    dcg = (rank_relevance * log_term).sum(axis=1)
    # compute IDCG
    # idcg is the ideal ranking, so all the relevant items must be at the top, namely all 1 have to be at the top
    idcg = np.array([(log_term[:min(int(n_pos), k)]).sum() for n_pos in ground_truth.sum(axis=1)])
    return dcg / idcg


def recall_at_k(pred_scores, ground_truth, k=10):
    """
    Computes the recall (at k) given the predicted scores and relevance of the items.
    :param pred_scores: score vector in output from the recommender (unsorted ranking)
    :param ground_truth: binary vector with relevance data (1 relevant, 0 not relevant)
    :param k: length of the ranking on which the metric has to be computed
    :return: recall at k position
    """
    k = min(pred_scores.shape[1], k)
    # generate ranking
    rank = np.argsort(-pred_scores, axis=1)
    # get relevance of first k items in the ranking
    rank_relevance = ground_truth[np.arange(pred_scores.shape[0])[:, np.newaxis], rank[:, :k]]
    # sum along axis 1 to count number of relevant items on first k-th positions
    # divide the number of relevant items in fist k positions by the number of relevant items to get recall
    return rank_relevance.sum(axis=1) / np.minimum(k, ground_truth.sum(axis=1))


def auc(pred_scores, ground_truth=None):
    """
    Computes the AUC of the given prediction scores.

    The given scores must be of shape n_examples x 2. The first score is the score for the positive item, the second
    score is the score for the negative item.

    The AUC counts the number of times that the positive item is ranked above the negative item.

    :param pred_scores: predicted scores for positive-negative pairs
    :return: AUC for each user
    """
    assert pred_scores.shape[1] == 2, "dim 1 must be of shape 2. There should be one positive and one negative item."
    return pred_scores[:, 1] > pred_scores[:, 0]


def mse(pred_scores, ground_truth):
    """
    Computes the Mean Squared Error between predicted and target ratings.

    :param pred_scores: predicted scores for validation user-item pairs
    :param ground_truth: target ratings for validation user-item pairs
    :return: Squared error for each user
    """
    assert pred_scores.shape == ground_truth.shape, "predictions and targets must match in shape."
    return np.square(pred_scores - ground_truth)


def fbeta(pred_scores, ground_truth, beta):
    """
    Computes the f beta measure between predictions and targets with the given beta value.

    :param pred_scores: predicted scores for validation user-item pairs
    :param ground_truth: target ratings for validation user-item pairs
    :return: fbeta measure
    """
    return fbeta_score(ground_truth, pred_scores, beta=beta,
                       pos_label=(0 if 0 in ground_truth else -1), average='binary')


def acc(pred_scores, ground_truth):
    """
    Computes the accuracy between predictions and targets.

    :param pred_scores: predicted scores for validation user-item pairs
    :param ground_truth: target ratings for validation user-item pairs
    :return: accuracy
    """
    return accuracy_score(ground_truth, pred_scores)


def isfloat(num):
    """
    Check if a string contains a float.
    :param num: string to be checked
    :return: True if num is float, False otherwise
    """
    try:
        float(num)
        return True
    except ValueError:
        return False


def check_metrics(metrics):
    """
    Check if the given list of metrics' names is correct.
    :param metrics: list of str containing the name of some metrics
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    assert all([isinstance(m, str) for m in metrics]), "The metrics must be represented as strings"
    # check all ranking based metrics
    assert all([m.split("@")[0] in valid_metrics for m in metrics if "@" in m]), "One of the selected ranking-based" \
                                                                                 " metrics is wrong."
    # check the k of ranking-based metrics is an integer
    assert all([m.split("@")[1].isdigit() for m in metrics if "@" in m]), "The k must be an integer"
    # check all the other metrics
    assert all([m in valid_metrics for m in metrics if "@" not in m if "-" not in m]), "Some of the given metrics are " \
                                                                                       "not valid. The accepted " \
                                                                                       "metrics are " + str(valid_metrics)
    # check fbeta
    assert all([m.split("-")[0] in valid_metrics for m in metrics if "-" in m]), "Some of the given metrics are " \
                                                                                       "not valid. The accepted " \
                                                                                       "metrics are " + str(valid_metrics)
    assert all([isfloat(m.split("-")[1]) for m in metrics if "-" in m]), "Some of the given metrics are " \
                                                                                 "not valid. The accepted " \
                                                                                 "metrics are " + str(valid_metrics)


def compute_metric(metric, pred_scores, ground_truth=None):
    """
    Compute the given metric on the given predictions and ground truth.
    :param metric: name of the metric that has to be computed
    :param pred_scores: score vector in output from the recommender (unsorted ranking)
    :param ground_truth: binary vector with relevance data (1 relevant, 0 not relevant)
    :return: the value of the given metric for the given predictions and relevance
    """
    if "@" in metric:
        m, k = metric.split("@")
        k = int(k)
        if m == "ndcg":
            return ndcg_at_k(pred_scores, ground_truth, k=k)
        if m == "hit":
            return hit_at_k(pred_scores, ground_truth, k=k)
        if m == "recall":
            return recall_at_k(pred_scores, ground_truth, k=k)
    elif "-" in metric:
        m, beta = metric.split("-")
        beta = float(beta)
        return fbeta(pred_scores, ground_truth, beta)
    else:
        if metric == "auc":
            return auc(pred_scores)
        if metric == "mse" or metric == "rmse":
            return mse(pred_scores, ground_truth)
        if metric == "acc":
            return acc(pred_scores, ground_truth)
