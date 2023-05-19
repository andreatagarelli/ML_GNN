"""
Input data load module.
-------------------------------------------

"""

import scipy as sp
import pandas as pd
import numpy as np
import dgl
import torch
from utils.util import sparse_mx_to_torch_sparse_tensor, normalize, standardize_features
from utils.data_generator import build_features
import os

MPX = 'MPX'
GML = 'GML'
DIRECTED = 'DIRECTED'
UNDIRECTED = 'UNDIRECTED'

DISTRIBUTION = ('mixed', 'gaussian', 'exponential', 'uniform', 'identity', 'one')


def load_dataset(dataset_name, root_dir, edges_name='net.edges'):
    n_entity, n_el, etype, tp = load_meta_information(root_dir)
    path = os.path.join(root_dir, edges_name)
    edges_mat = pd.read_csv(filepath_or_buffer=path, sep=' ', header=None).to_numpy(dtype=np.int32)

    if tp == MPX:
        edges_mat = edges_mat[:, 0:3]

    return edges_mat, n_entity, n_el, etype, tp


def load_meta_information(root_dir, meta_info='meta_info.txt'):
    path = os.path.join(root_dir, meta_info)
    minfo = pd.read_csv(filepath_or_buffer=path, sep=' ')
    minfo.columns = minfo.columns.str.strip().str.upper()
    n_entity = int(minfo['N'][0])
    tp = minfo['TYPE'][0]
    assert tp == MPX or tp == GML, 'Assertion Error: Unrecognized Network type.'
    etype = minfo['E'][0]
    assert etype == DIRECTED \
           or etype == UNDIRECTED, 'Assertion Error: Unrecognized edge type.'
    num_layers = int(minfo['L'][0])

    return n_entity, num_layers, etype, tp


def load_labels(dataset_name, root_dir, labels_name='nodes.txt'):
    path = os.path.join(root_dir, labels_name)
    #print(f'Loading labels from {path}')
    if dataset_name == 'dkpol' or dataset_name == 'aucs':  # Not numerical labels
        sep = ','
    else:
        sep = ' '

    labels = pd.read_csv(filepath_or_buffer=path, sep=sep, header=None)[0].to_numpy(dtype=np.dtype(str))

    return labels


def encode_onehot(labels):
    """ One hot-encoding of labels """

    classes = np.unique(labels)  # unique sorted values
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def build_supra_adj(edges, nol, n_entity, etype, ntype, add_self_loop=True,
                    add_inter_layer_edges=True):

    """Build supra-adjacency matrix"""
    dim = n_entity * nol
    if ntype == MPX:
        le = edges[edges[:, 0] == 1][:, 1:3]
        rows = le[:, 0]
        cols = le[:, 1]

        for i in range(2, nol + 1):
            le = edges[edges[:, 0] == i][:, 1:3]
            le = le + n_entity*(i-1)
            rows = np.concatenate((rows, le[:, 0]))
            cols = np.concatenate((cols, le[:, 1]))

    elif ntype == GML:
        rows, cols = None, None
        for i in range(1, nol + 1):
            for j in range(1, nol + 1):
                le = edges[(edges[:, 0] == i) & (edges[:, 2] == j)]
                if rows is None:
                    rows = le[:, 1] + (n_entity * (i - 1))
                    cols = le[:, 3] + (n_entity * (j - 1))
                else:

                    nr = le[:, 1] + (n_entity * (i - 1))
                    nc = le[:, 3] + (n_entity * (j - 1))
                    rows = np.concatenate((rows, nr))
                    cols = np.concatenate((cols, nc))
    else:
        raise ValueError('Unrecognized network type.')

    adj = sp.sparse.csr_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(dim, dim))
    empty = np.ones(dim, dtype=bool)
    empty[np.unique(adj.nonzero())] = False

    if add_self_loop:
        adj = adj + sp.sparse.identity(dim, format='csr')  # add_self_loop

    if add_inter_layer_edges and ntype == MPX:
        data = 1
        offset = []
        for i in range(1, nol):
            x = i * n_entity
            mx = -i * n_entity
            offset.append(x)
            offset.append(mx)

        inter_adj = sp.sparse.diags([data]*len(offset), offset, shape=(dim, dim), format='csr')
        adj = adj + inter_adj
    if (add_self_loop or add_inter_layer_edges) and ntype == MPX:
        # cleaning
        adj = adj.tolil()
        adj[empty] = .0
        adj[:, empty] = .0
        adj = adj.tocsr()

    if etype == UNDIRECTED:
        # make the supra-adjacency matrix symmetric
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, empty


def build_DGLgraph(adj, empty=None, add_self_loop=False, to_bidirected=False):
    """
    Build dgl.DGLGraph from supra-adjacency matrix
    """
    g = dgl.DGLGraph(adj)
    if empty is not None:
        g.ndata['empty'] = torch.tensor(empty)
        if add_self_loop:
            real = ~empty
            g.add_edges(g.nodes()[real], g.nodes()[real])
    elif add_self_loop:
        g = dgl.add_self_loop(g)
    if to_bidirected:
        g = dgl.to_bidirected(g)
    return g


def build_graph(dataset, data, add_inter_layer_edges=True, add_self_loop=True):
    """
    Build the input DGL graph
    """
    edges, n_entity, n_el, etype, tp = load_dataset(dataset, data)
    adj, empty = build_supra_adj(edges, n_el, n_entity, etype, add_self_loop=add_self_loop,
                                 add_inter_layer_edges=add_inter_layer_edges,
                                 ntype=tp)
    g = build_DGLgraph(adj, empty=empty)
    return g, n_entity, n_el


def train_test_val_split(labels, train_percentage, test_percentage, num_classes, shuffle=True):
    """ Builds the indices of training, test and validation nodes """
    labi = np.where(labels == 0)[0]
    if labi.size == 0:
        raise ValueError('Error! Labels must start from zero.')
    np.random.shuffle(labi)
    k = len(labi)

    val_percentage = 100 - (train_percentage + test_percentage)
    num_train = int(np.ceil(k * train_percentage/100))

    idx_train = labi[0:num_train]
    if val_percentage <= 0:
        idx_test = labi[num_train:]
    else:
        num_test = int(np.ceil(k * test_percentage/100))
        idx_test = labi[num_train:num_train + num_test]
        idx_val = labi[num_train + num_test:]
    for i in range(1, num_classes):
        labi = np.where(labels == i)[0]
        np.random.shuffle(labi)
        k = len(labi)
        num_train = int(np.ceil(k * train_percentage/100))
        idx_train = np.concatenate((idx_train, labi[0:num_train]))
        if val_percentage <= 0:
            idx_test = np.concatenate((idx_test, labi[num_train:]))

        else:
            num_test = int(np.ceil(k * test_percentage/100))
            idx_test = np.concatenate((idx_test, labi[num_train:num_train + num_test]))
            idx_val = np.concatenate((idx_val, labi[num_train + num_test:]))
    if val_percentage <= 0:
        idx_val = idx_test
    if shuffle:
        np.random.shuffle(idx_train)
        np.random.shuffle(idx_test)
        np.random.shuffle(idx_val)

    return idx_train, idx_test, idx_val


def load_data(root_dir, dataset_name, num_features, train_percentage, test_percentage,
              features_distribution='gaussian', add_inter_layer_edges=True, feat_var='fixed',
              standardize=True, normz=False):
    """
    Load input graph, features, labelse, train indices, validation indices,
    test indices, input dataset information.
    """
    dataset_name = dataset_name.lower()
    data = os.path.join(root_dir, dataset_name)
    g, n, n_el = build_graph(dataset_name, data=data, add_self_loop=True,
                             add_inter_layer_edges=add_inter_layer_edges)

    labels = encode_onehot(load_labels(dataset_name, root_dir=data))
    num_classes = labels.shape[1]
    labels = np.where(labels)[1]
    idx_train, idx_test, idx_val = train_test_val_split(labels, train_percentage, test_percentage, num_classes)
    if features_distribution.lower() not in DISTRIBUTION:
        if not features_distribution.endswith('.csv'):
            features_distribution = features_distribution + ".csv"
        path = os.path.join(data, features_distribution)
        #print(f"Loading real world features for {dataset_name} dataset from {path} ")
        df = pd.read_csv(filepath_or_buffer=path, sep=',')
        if standardize:
            #print('Real world features have been standardized.')
            af = standardize_features(df.values)
        elif normz:
            #print('Real world features row-wise have been normalized.')
            af = normalize(df.values)
        else:
            af = df.values

        af = sp.sparse.csr_matrix(af)
        #print('Features have been assigned to each layer.')
        af = sp.sparse.vstack(np.repeat(af, n_el))
        features = sparse_mx_to_torch_sparse_tensor(af)

    else:
        features = build_features(n_entity=n, nol=n_el, num_features=num_features,
                                  distribution=features_distribution.lower(), empty=g.ndata['empty'],
                                  mode=feat_var.lower())

    g.ndata.pop('empty')
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    d_info = (n, [n_el])
    return g, features, labels, idx_train, idx_val, idx_test, d_info

