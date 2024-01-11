import torch
from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F


class ATTNet(nn.Module):
    def __init__(self, num_users, num_hidden_layers, hidden_features, out_features=1,
                 outermost_linear='sigmoid', nonlinearity='relu', use_profile=False):
        super(ATTNet, self).__init__()

        nls = {'relu': nn.ReLU(inplace=True),
               'sigmoid': nn.Sigmoid(),
               'tanh': nn.Tanh(),
               'selu': nn.SELU(inplace=True),
               'softplus': nn.Softplus(),
               'elu': nn.ELU(inplace=True)}

        nl = nls[nonlinearity]
        nl_outermost = nls[outermost_linear]

        self.use_profile = use_profile
        if use_profile:
            self.embed_profiles = []
            self.embed_profiles.append(nn.Sequential(
                nn.Linear(768, hidden_features), nl
            ))
            self.embed_profiles = nn.Sequential(*self.embed_profiles)

            self.embed_att = []
            self.embed_att.append(nn.Sequential(
                nn.Linear(hidden_features, 1), nn.Softmax()
            ))
            self.embed_att = nn.Sequential(*self.embed_att)

        self.hidden_features = hidden_features

        self.embed_users = []
        self.embed_users.append(nn.Sequential(
            nn.Embedding(num_users, hidden_features), nl
        ))
        for i in range(num_hidden_layers - 1):
            self.embed_users.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))
        self.embed_users = nn.Sequential(*self.embed_users)

        self.embed_times = []
        self.embed_times.append(nn.Sequential(
            nn.Linear(1, hidden_features), nl
        ))
        for i in range(num_hidden_layers - 1):
            self.embed_times.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))
        self.embed_times = nn.Sequential(*self.embed_times)

        self.net = []
        if use_profile:
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features * 3, hidden_features), nl
            ))
        else:
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))
        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))
        self.net.append(nn.Sequential(
            nn.Linear(hidden_features, out_features), nl_outermost
        ))
        self.net = nn.Sequential(*self.net)
        self.hidden_size = 64
        self.long_w = nn.Linear(1, self.hidden_size)
        self.long_short_w = nn.Linear(hidden_features, hidden_features)
        self.long_fusion_w = nn.Linear(hidden_features, hidden_features)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.layerNorm = nn.LayerNorm(hidden_features, eps=1e-12)
        self.concatlayer = nn.Linear(hidden_features * 2, hidden_features)
        self.lstm = nn.LSTM(input_size=hidden_features,
                            hidden_size=hidden_features,
                            batch_first=True,
                            bidirectional=False)
        self.ui_linear = nn.Linear(hidden_features, hidden_features, bias=False)
        self.ti_linear = nn.Linear(1, hidden_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_features, ))
        self.relu = nn.ReLU()
        self.out_layer = nn.Linear(hidden_features,1)
        self.out_layer_1 = nn.Linear(hidden_features,1)
    def forward(self, times, users, profs=None, params=None, **kwargs):

        x = self.embed_times(times.float())
        y = self.embed_users(users.long())

        combined = self.sigmoid(self.ti_linear(x.unsqueeze(-1)) + self.ui_linear(y) + self.bias)

        mask_all = combined.data.eq(0)
        x_u = self.attention(combined, mask_all)
        x_u = self.tanh(x_u + combined)


        h_u, _ = self.lstm(x_u)
        output = self.tanh(self.out_layer_1(self.out_layer(h_u).squeeze()))


        return output

    def attention(self, seq_opinion_embedding, mask=None):
        """

        get the long term purpose of user
        """
        seq_opinion_embedding = seq_opinion_embedding.float()
        seq_opinion_embedding_value = seq_opinion_embedding
        seq_opinion_embedding = self.relu(self.long_short_w(seq_opinion_embedding))
        # batch_size * seq_len
        if mask is not None:
            seq_opinion_embedding.masked_fill_(mask, -1e9)
        x_u = nn.Softmax(dim=-1)(seq_opinion_embedding)
        x_u = torch.mul(seq_opinion_embedding_value,
                                        x_u)
        # batch_size * 1 * embedding_size

        return x_u

