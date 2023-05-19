"""
Cross-layer aggregation module.
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


class FusionLayer(nn.Module):
    def __init__(self, in_dim, num_classes, nol, ff_layers=1, trainable_fweights=True, fusion_weights=None,
                 activation=None, bias=True):

        super(FusionLayer, self).__init__()
        self.d = in_dim
        self.fc_layers = nn.ModuleList()
        for i in range(ff_layers - 1):
            self.fc_layers.append(nn.Linear(in_dim, in_dim, bias=bias))
        self.fc_layers.append(nn.Linear(in_dim, num_classes, bias=bias))

        if trainable_fweights:
            self.fusion = nn.Parameter(torch.FloatTensor(size=(nol, 1, 1)))  # Trainable aggregation parameters
        elif not fusion_weights:
            self.fusion = nn.Parameter(torch.FloatTensor(size=(nol, 1, 1)), requires_grad=False)

        self.activation = activation
        self.nol = nol
        self.hp = None
        self.reset_parameters(fusion_weights=fusion_weights)

    def reset_parameters(self, fusion_weights=None):

        gain = nn.init.calculate_gain('relu')
        for m in self.fc_layers:
            nn.init.xavier_normal_(m.weight, gain=gain)

        if fusion_weights:
            self.fusion = nn.Parameter(torch.FloatTensor(fusion_weights).view(self.nol, 1, 1), requires_grad=False)
        else:
            nn.init.constant_(self.fusion, val=1/self.nol)

    def forward(self, h):

        self.hp = (h.view(self.nol, -1, self.d) * self.fusion).sum(0)
        h = self.hp
        for i in range(len(self.fc_layers)-1):
            h = F.relu(self.fc_layers[i](h))
        logits = self.fc_layers[-1](h)
        logits = self.activation(logits, dim=1)
        return logits
