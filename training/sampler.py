import torch

from torch.utils.data import Sampler
from utils.util import build_batch_entities


class CustomSampler(Sampler):

    def __init__(self, data_source, n_train, nol, entity_batch_size, shuffle=True):
        """
        Sampling for Multilayer Network
        :param data_source: dataset to sample from
        :param n_train: n_train == len(data_source)//batch_size
        :param nol: number of layers
        :param entity_batch_size: batch size at entity level
        :param shuffle: Whether to shuffle the training nodes
        """
        if not isinstance(shuffle, bool):
            raise ValueError("Shuffle should be a boolean value, but got "
                             "shuffle={}".format(shuffle))

        self.data_source = data_source
        self.n_train = n_train
        self.nol = nol
        self.entity_batch_size = entity_batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if not self.shuffle:
            return iter(range(len(self.data_source)))

        rand_perm = torch.randperm(self.n_train)  # random permutation for training entities
        _, rand_perm = build_batch_entities(rand_perm, batch=self.entity_batch_size, l=self.nol, n=self.n_train, shuffle=False,
                                            sort=False)

        return iter(rand_perm.tolist())

    def __len__(self):
        return len(self.data_source)

