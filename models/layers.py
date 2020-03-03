# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from utils import config
from models.basic import BasicModule
from models.attention import Attention


class Encoder(BasicModule):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_word_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

        self.init_params()

    # seq_lens should be in descending order
    def forward(self, input, seq_lens):
        embedded = self.src_word_emb(input)

        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x l x n
        encoder_outputs = encoder_outputs.contiguous()

        encoder_feature = encoder_outputs.view(-1, 2 * config.hidden_dim)   # B*l x 2*hidden_dim
        encoder_feature = self.fc(encoder_feature)

        return encoder_outputs, encoder_feature, hidden


class ReduceState(BasicModule):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.init_params()


    def forward(self, hidden):
        h, c = hidden  # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))  # h, c dim = 1 x b x hidden_dim

class Decoder(BasicModule):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()
        # decoder
        self.tgt_word_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.con_fc = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, batch_first=True, bidirectional=False)

        if config.pointer_gen:
            self.p_gen_fc = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        # p_vocab
        self.fc1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.vocab_size)

        self.init_params()

    def forward(self, y_t, s_t, enc_out, enc_fea, enc_padding_mask,
                c_t, extra_zeros, enc_batch_extend_vocab, coverage, step):

        if not self.training and step == 0:
            dec_h, dec_c = s_t
            s_t_hat = torch.cat((dec_h.view(-1, config.hidden_dim),
                                 dec_c.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, enc_out, enc_fea,
                                                           enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_embd = self.tgt_word_emb(y_t)
        x = self.con_fc(torch.cat((c_t, y_t_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t)

        dec_h, dec_c = s_t
        s_t_hat = torch.cat((dec_h.view(-1, config.hidden_dim),
                             dec_c.view(-1, config.hidden_dim)), 1)     # B x 2*hidden_dim
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, enc_out, enc_fea,
                                                               enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_inp = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_fc(p_gen_inp)
            p_gen = torch.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1)  # B x hidden_dim * 3
        output = self.fc1(output)  # B x hidden_dim
        # output = F.relu(output)

        output = self.fc2(output)  # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage
