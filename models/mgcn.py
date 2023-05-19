import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
from models.fusion_layer import FusionLayer


class MGCN(nn.Module):

    def __init__(self, in_dim, num_hidden, num_classes, num_layers, activation, n_el,
                 dropout=.0, trainable_fweights=True, fusion_weights=None, ff_layers=1):
        super(MGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_dim, num_hidden, norm='both', activation=activation, allow_zero_in_degree=True,
                                     bias=True))

        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, allow_zero_in_degree=True,
                                         bias=True))
        # output projection
        self.layers.append(FusionLayer(num_hidden, num_classes, n_el, activation=F.log_softmax,
                                       trainable_fweights=trainable_fweights,
                                       fusion_weights=fusion_weights, ff_layers=ff_layers))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, blocks, inputs):
        h = inputs
        for i in range(len(self.layers)-1):
            if i != 0:
                h = self.dropout(h)
            layer = self.layers[i]
            h = layer(blocks[i], h)

        logits = self.layers[-1](h)
        return logits

    def inference(self, g, inputs):
        h = inputs
        for i in range(len(self.layers)-1):
            if i != 0:
                h = self.dropout(h)
            layer = self.layers[i]
            h = layer(g, h)
        logits = self.layers[-1](h)
        return logits
