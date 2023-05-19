
import scipy as sp
import numpy as np
import torch
from utils.util import sparse_mx_to_torch_sparse_tensor


DISTRIBUTION = ('mixed', 'gaussian', 'exponential', 'uniform', 'identity', 'one')


def generate_features(n_entity, num_features, distribution='mixed'):
    if distribution == 'mixed':
        g = int(num_features/3)
        x1 = np.random.normal(size=(g, n_entity)).T
        x2 = np.random.exponential(size=(g, n_entity)).T
        x3 = np.random.uniform(size=(num_features-g*2, n_entity)).T
        return np.hstack((x1, x2, x3))
    elif distribution == 'gaussian':
        return np.random.normal(size=(num_features, n_entity)).T
    elif distribution == 'exponential':
        return np.random.exponential(size=(num_features, n_entity)).T
    elif distribution == 'uniform':
        return np.random.uniform(size=(num_features, n_entity)).T

    else:
        raise ValueError('Unrecognized distribution')


def build_features(n_entity, nol, num_features, empty, distribution='mixed', mode='layer'):

    if mode not in ('fixed', 'layer'):
        raise ValueError('Unrecognized feature matrix creation mode.')
    if distribution not in DISTRIBUTION:
        raise ValueError('Unrecognized distribution.')
    if mode == 'layer' and (distribution == 'one' or distribution == 'identity'):
        raise ValueError('Features cannot be changed between layers.')

    if distribution == 'one':
        distr = torch.ones(n_entity*nol).view(-1, 1)
        distr[empty] = .0
        return distr

    if distribution == 'identity':
        print('Setting features off!')
        data = 1
        offset = []
        for i in range(0, nol):
            x = -i * n_entity
            offset.append(x)
        dim = n_entity*nol
        a = sp.sparse.diags([data] * len(offset), offset, shape=(dim, n_entity), format='csr')
        return sparse_mx_to_torch_sparse_tensor(a)

    x = generate_features(n_entity, num_features, distribution=distribution)
    for i in range(nol):
        eli = empty[n_entity*i: n_entity*(i+1)].numpy()  # empty layer i
        if mode == 'layer':
            y = x + np.random.standard_normal(size=(n_entity, num_features))  # Add Gaussian noise
        elif mode == 'fixed':
            y = x
        if i == 0:
            a = sp.sparse.lil_matrix(y)
            a[eli] = .0
        else:
            b = sp.sparse.lil_matrix(y)
            b[eli] = .0
            a = sp.sparse.vstack((a, b), format='csr')

    return sparse_mx_to_torch_sparse_tensor(a)
