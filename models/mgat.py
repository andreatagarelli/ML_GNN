
import torch.nn as nn
from dgl.nn.pytorch import GATConv
import torch.nn.functional as F
from models.fusion_layer import FusionLayer


class MGAT(nn.Module):
    _multi_head = {'concat', 'avg'}  # Multi-head attention strategy

    def __init__(self, num_layers, in_dim, num_hidden, num_classes, heads, activation, n_el, aggregation='concat',
                 drop=.0, attn_drop=.0, negative_slope=0.2, residual=False, trainable_fweights=True,
                 fusion_weights=None, ff_layers=1):

        super(MGAT, self).__init__()
        aggregation = aggregation.lower()
        if aggregation not in self._multi_head:
            raise ValueError("Unrecognized  aggregation mode of Q attention heads : {} ".format(aggregation))
        self.aggregation = aggregation
        self.num_layers = num_layers
        self.activation = activation
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            .0, attn_drop, negative_slope, residual=residual, activation=self.activation, allow_zero_in_degree=True,
            bias=True))
        for l in range(1, num_layers):
            if self.aggregation == 'concat':
                # in_dim = num_hidden * num_heads due to concatenation of the attention heads
                inf = num_hidden * heads[l-1]
            else:
                inf = num_hidden
            self.layers.append(GATConv(
                inf, num_hidden, heads[l],
                drop, attn_drop, negative_slope, residual, self.activation, allow_zero_in_degree=True, bias=True))

        if self.aggregation == 'concat':
            inf = num_hidden*heads[-1]
        else:
            inf = num_hidden

        # Output projection
        self.layers.append(FusionLayer(in_dim=inf, num_classes=num_classes, nol=n_el,
                                       activation=F.log_softmax, trainable_fweights=trainable_fweights,
                                       fusion_weights=fusion_weights, ff_layers=ff_layers))

    def forward(self, blocks, inputs):
        h = inputs
        for i in range(self.num_layers):
            if self.aggregation == 'concat':
                h = self.layers[i](blocks[i], h).flatten(1)
            else:
                h = self.layers[i](blocks[i], h).mean(1)
        logits = self.layers[-1](h)  # output layer (cross-layer aggregation)
        return logits

    def inference(self, g, inputs):
        h = inputs
        for i in range(self.num_layers):
            if self.aggregation == 'concat':
                h = self.layers[i](g, h).flatten(1)
            else:
                h = self.layers[i](g, h).mean(1)
        logits = self.layers[-1](h)  # output layer (cross-layer aggregation)
        return logits
