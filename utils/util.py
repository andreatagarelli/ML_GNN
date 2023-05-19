"""
Utility functions
"""
import torch
import numpy as np
import scipy as sp
import math
from sklearn.metrics import f1_score, precision_score, recall_score, \
     label_ranking_average_precision_score, classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


def accuracy(labels, preds):
    """
    Compute accuracy
    :param labels: torch.Tensor
    :param preds:  torch.Tensor
    :return: float
        accuracy score.
    """
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accscore(labels, preds):
    return accuracy_score(labels, preds)


def prediction(output):
    return output.max(1)[1]


def recall(labels, preds, average='micro'):
    """
    Compute recall
    :param labels: torch.Tensor
    :param preds: torch.Tensor
    :param average: str
    :return: recall score
    """
    return recall_score(labels, preds, average=average)


def precision(labels, preds, average='micro'):
    """
    Compute precision
    :param labels: torch.Tensor
    :param preds: torch.Tensor
    :param average: str
    :return: precision
    """
    return precision_score(labels, preds, average=average)


def fmeasure(labels, preds, average='micro'):
    """
    Compute F1-score
    :param labels: torch.Tensor
    :param preds: torch.Tensor
    :param average: str
    :return: F1 score
    """
    return f1_score(labels, preds, average=average)


def confusion_mat(labels, preds, classes=None):
    return confusion_matrix(labels, preds, labels=classes)


def report(labels, preds):
    return classification_report(labels, preds)


def rlap(labels, output):
    """
    Rank Label average precision
    :param labels:
    :param output:
    :return:
    """
    return label_ranking_average_precision_score(labels, output)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Ref : https://github.com/tkipf/pygcn
    :param sparse_mx: scipy.sparse
    :return:
    """

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
                                np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    """
    Row-normalization
    Ref: https://github.com/tkipf/pygcn
    :param mx: scipy.sparse
    :return: scipy.sparse

    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def standardize_features(f):
    scaler = StandardScaler()
    scaler.fit(f)
    return scaler.transform(f)


def build_train_nids(idx_train, l, n):
    for i in range(l):
        if i == 0:
            train_nids = idx_train
        else:
            train_nids = torch.cat((train_nids, idx_train + i * n), axis=0)
    return train_nids


def build_batch_entities(idx_train, batch, l, n, shuffle=False, sort=False):
    if batch <= 0:
        raise ValueError('Batch size less equal than zero.')

    if shuffle:
        idx_train = idx_train[torch.randperm(len(idx_train))]
    elif sort:
        idx_train = idx_train.sort()[0]
    if batch >= len(idx_train):
        return int(batch*l), build_train_nids(idx_train, l, n)
    else:
        l_it = math.ceil(n/batch)
        for i in range(l_it):
            idx = idx_train[i*batch:(i+1)*batch]
            temp = build_train_nids(idx, l, n)
            if i == 0:
                newidx_train = temp
            else:
                newidx_train = torch.cat((newidx_train, temp), axis=0)

        return int(batch*l), newidx_train


def expand_index(idx, layers, n, sort=True):
    if sort:
        idx = idx.sort()[0]
    for i in range(layers):
        if i == 0:
            n_idx = idx
        else:
            n_idx = torch.cat((n_idx, idx + i * n), axis=0)
    return n_idx
