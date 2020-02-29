# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import config
from models.basic import BasicModule


class Attention(BasicModule):
    def __init__(self):
        super(Attention, self).__init__()

        self.fc = nn.Linear(config.hidden_dim * 2, 1, bias=False)
        self.dec_fc = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        if config.is_coverage:
            self.con_fc = nn.Linear(1, config.hidden_dim * 2, bias=False)

        self.init_params()

    def forward(self, s_t, enc_out, enc_fea, enc_padding_mask, coverage):
        b, l, n = list(enc_out.size())

        dec_fea = self.dec_fc(s_t)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, l, n).contiguous()  # B x l x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)     # B*l x 2*hidden_dim

        att_features = enc_fea + dec_fea_expanded           # B*l x 2*hidden_dim
        if config.is_coverage:
            coverage_inp = coverage.view(-1, 1)             # B*l x 1
            coverage_fea = self.con_fc(coverage_inp)        # B*l x 2*hidden_dim
            att_features = att_features + coverage_fea

        e = torch.tanh(att_features)                        # B*l x 2*hidden_dim
        scores = self.fc(e)                                 # B*l x 1
        scores = scores.view(-1, l)                         # B x l

        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask  # B x l
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)                        # B x 1 x l
        c_t = torch.bmm(attn_dist, enc_out)                       # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)                 # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, l)                         # B x l

        if config.is_coverage:
            coverage = coverage.view(-1, l)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage

