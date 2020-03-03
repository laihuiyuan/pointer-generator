# -*- coding: utf-8 -*-

import torch

from utils import config
from numpy import random
from models.layers import Encoder
from models.layers import Decoder
from models.layers import ReduceState
from transformer.model import TranEncoder

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class Model(object):
    def __init__(self, model_path=None, is_eval=False, is_tran = False):
        encoder = Encoder()
        decoder = Decoder()
        reduce_state = ReduceState()
        if is_tran:
            encoder = TranEncoder(config.vocab_size, config.max_enc_steps, config.emb_dim,
                 config.n_layers, config.n_head, config.d_k, config.d_v, config.d_model, config.d_inner)

        # shared the embedding between encoder and decoder
        decoder.tgt_word_emb.weight = encoder.src_word_emb.weight

        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()

        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

        if model_path is not None:
            state = torch.load(model_path, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])
