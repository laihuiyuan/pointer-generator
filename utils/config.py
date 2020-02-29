# -*- coding: utf-8 -*-

import os

beam_size=4
emb_dim= 128
batch_size= 16
hidden_dim= 256
max_enc_steps=400
max_dec_steps=120
max_tes_steps=100
min_dec_steps=35
vocab_size=50000

lr=0.15
cov_loss_wt = 1.0
pointer_gen = True
is_coverage = True

max_grad_norm=2.0
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4

eps = 1e-12
use_gpu=True
lr_coverage=0.15
max_iterations = 500000

SENTENCE_STA = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
BOS_TOKEN = '[BOS]'
EOS_TOKEN = '[EOS]'

root_dir = os.path.expanduser("../")
log_root = os.path.join(root_dir, "pointer_summarizer/dataset/log/")

#train_data_path = os.path.join(root_dir, "pointer_summarizer/dataset/finished_files/train.bin")
train_data_path = os.path.join(root_dir, "pointer_summarizer/dataset/finished_files/chunked/train_*")
eval_data_path = os.path.join(root_dir, "pointer_summarizer/dataset/finished_files/val.bin")
decode_data_path = os.path.join(root_dir, "pointer_summarizer/dataset/finished_files/test.bin")
vocab_path = os.path.join(root_dir, "pointer_summarizer/dataset/finished_files/vocab")
