# -*- coding: utf-8 -*-

import os
import pyrouge
import logging
import numpy as np

import torch
import tensorflow as tf
from torch.autograd import Variable

from utils import config


def article2ids(article_words, vocab):
    ids = []
    oov = []
    unk_id = vocab.word2id(config.UNK_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is OOV
            if w not in oov:  # Add to list of OOVs
                oov.append(w)
            oov_num = oov.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oov


def abstract2ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word2id(config.UNK_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, \
                "Error: models produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError(
                    'Error: models produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                        i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


def abstract2sents(abstract):
    cur_p = 0
    sents = []
    while True:
        try:
            sta_p = abstract.index(config.SENTENCE_STA.encode(), cur_p)
            end_p = abstract.index(config.SENTENCE_END.encode(), sta_p + 1)
            cur_p = end_p + len(config.SENTENCE_END.encode())
            sents.append(abstract[sta_p + len(config.SENTENCE_STA.encode()):end_p])
        except ValueError as e:  # no more sentences
            return sents


def show_art_oovs(article, vocab):
    unk_token = vocab.word2id(config.UNK_TOKEN)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w) == unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
    unk_token = vocab.word2id(config.UNK_TOKEN)
    words = abstract.split(' ')
    new_words = []
    for w in words:
        if vocab.word2id(w) == unk_token:  # w is oov
            if article_oovs is None:  # baseline mode
                new_words.append("__%s__" % w)
            else:  # pointer-generator mode
                if w in article_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else:  # w is in-vocab word
            new_words.append(w)
    out_str = ' '.join(new_words)
    return out_str


def print_results(article, abstract, decoded_output):
    print("")
    print('ARTICLE:  %s', article)
    print('REFERENCE SUMMARY: %s', abstract)
    print('GENERATED SUMMARY: %s', decoded_output)
    print("")


def make_html_safe(s):
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def rouge_eval(ref_dir, dec_dir):
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
    log_str = ""
    for x in ["1", "2", "l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x, y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    print(log_str)
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    print("Writing final ROUGE results to %s..." % (results_file))
    with open(results_file, "w") as f:
        f.write(log_str)


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
    if running_avg_loss == 0:  # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    return running_avg_loss


def write_for_rouge(reference_sents, decoded_words, ex_index,
                    _rouge_ref_dir, _rouge_dec_dir):
    decoded_sents = []
    while len(decoded_words) > 0:
        try:
            fst_period_idx = decoded_words.index(".")
        except ValueError:
            fst_period_idx = len(decoded_words)
        sent = decoded_words[:fst_period_idx + 1]
        decoded_words = decoded_words[fst_period_idx + 1:]
        decoded_sents.append(' '.join(sent))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    reference_sents = [make_html_safe(w) for w in reference_sents]

    ref_file = os.path.join(_rouge_ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(_rouge_dec_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w") as f:
        for idx, sent in enumerate(reference_sents):
            f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
    with open(decoded_file, "w") as f:
        for idx, sent in enumerate(decoded_sents):
            f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")

    # print("Wrote example %i to file" % ex_index)


def get_input_from_batch(batch, use_cuda):
    extra_zeros = None
    enc_lens = batch.enc_lens
    max_enc_len = np.max(enc_lens)
    enc_batch_extend_vocab = None
    batch_size = len(batch.enc_lens)
    enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
    enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()

    if config.pointer_gen:
        enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
        # max_art_oovs is the max over all the article oov list in the batch
        if batch.max_art_oovs > 0:
            extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))

    c_t = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))

    coverage = None
    if config.is_coverage:
        coverage = Variable(torch.zeros(enc_batch.size()))

    enc_pos = np.zeros((batch_size, max_enc_len))
    for i, inst in enumerate(batch.enc_batch):
        for j, w_i in enumerate(inst):
            if w_i != config.PAD:
                enc_pos[i, j] = (j + 1)
            else:
                break
    enc_pos = Variable(torch.from_numpy(enc_pos).long())

    if use_cuda:
        c_t = c_t.cuda()
        enc_pos = enc_pos.cuda()
        enc_batch = enc_batch.cuda()
        enc_padding_mask = enc_padding_mask.cuda()

        if coverage is not None:
            coverage = coverage.cuda()

        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()

        if enc_batch_extend_vocab is not None:
            enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()


    return enc_batch, enc_lens, enc_pos, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, c_t, coverage


def get_output_from_batch(batch, use_cuda):
    dec_lens = batch.dec_lens
    max_dec_len = np.max(dec_lens)
    batch_size = len(batch.dec_lens)
    dec_lens = Variable(torch.from_numpy(dec_lens)).float()
    tgt_batch = Variable(torch.from_numpy(batch.tgt_batch)).long()
    dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
    dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()

    dec_pos = np.zeros((batch_size, config.max_dec_steps))
    for i, inst in enumerate(batch.dec_batch):
        for j, w_i in enumerate(inst):
            if w_i != config.PAD:
                dec_pos[i, j] = (j + 1)
            else:
                break
    dec_pos = Variable(torch.from_numpy(dec_pos).long())

    if use_cuda:
        dec_lens = dec_lens.cuda()
        tgt_batch = tgt_batch.cuda()
        dec_batch = dec_batch.cuda()
        dec_padding_mask = dec_padding_mask.cuda()
        dec_pos = dec_pos.cuda()

    return dec_batch, dec_lens, dec_pos, dec_padding_mask, max_dec_len, tgt_batch
