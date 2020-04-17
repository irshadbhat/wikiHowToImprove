from __future__ import unicode_literals

import io
import re
import sys
import time
import random
import pickle

from argparse import ArgumentParser
from collections import Counter, defaultdict
from sklearn.metrics import classification_report as cr

import gensim
import dynet as dy
import numpy as np

import torch
from pytorch_transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader

class Meta:
    def __init__(self):
        self.c_dim = 32  # character-rnn input dimension
        self.w_dim_e = 0  # pretrained word embedding size (0 if no pretrained embeddings)
        self.n_hidden = 64  # pos-mlp hidden layer dimension
        self.lstm_char_dim = 32  # char-LSTM output dimension
        self.lstm_word_dim = 128  # LSTM (word-char concatenated input) output dimension


class Tagger():
    def __init__(self, model=None, meta=None, wvm=None):
        self.model = dy.Model()
        self.meta = pickle.load(open('%s.meta' %model, 'rb')) if model else meta
        self.trainer = self.meta.trainer(self.model)
        # pretrained embeddings
        if wvm:
            self.wvm = wvm
            self.meta.w_dim_e = wvm.syn0.shape[1]

        # MLP on top of biLSTM outputs 100 -> 32 -> ntags
        self.w1 = self.model.add_parameters((self.meta.n_hidden, self.meta.lstm_word_dim*2))
        self.b1 = self.model.add_parameters(self.meta.n_hidden)
        self.vt = self.model.add_parameters((1, self.meta.n_hidden))

        self.aw = self.model.add_parameters((self.meta.lstm_word_dim, self.meta.lstm_word_dim*2))
        self.ab = self.model.add_parameters(self.meta.lstm_word_dim)
        self.av = self.model.add_parameters((1, self.meta.lstm_word_dim))

        # word-level LSTMs
        self.fwdRNN = dy.LSTMBuilder(1, self.meta.w_dim_e+self.meta.lstm_char_dim*0, self.meta.lstm_word_dim, self.model) 
        self.bwdRNN = dy.LSTMBuilder(1, self.meta.w_dim_e+self.meta.lstm_char_dim*0, self.meta.lstm_word_dim, self.model)
        self.fwdRNN2 = dy.LSTMBuilder(1, self.meta.lstm_word_dim*2, self.meta.lstm_word_dim, self.model) 
        self.bwdRNN2 = dy.LSTMBuilder(1, self.meta.lstm_word_dim*2, self.meta.lstm_word_dim, self.model)

        # unk for unknown word embeddings
        self.unk = np.zeros(self.meta.w_dim_e)

        # load pretrained dynet model
        if model:
            self.model.populate('%s.dy' %model)

    def word_rep(self, batch):
        #char_embs = self.char_rep([w for wi in zip(*[b[0] for b in batch]) for w in wi])
        batch_embs1 = [[] for _ in range(len(batch[0][0]))]
        batch_embs2 = [[] for _ in range(len(batch[0][0]))]
        for s1, s2 in batch:
            for i,(word1, word2) in enumerate(zip(s1, s2)):
                if word1 == 'PAD' or word1 not in self.wvm:
                    batch_embs1[i].append(self.unk)
                else:
                    batch_embs1[i].append(self.wvm[word1])
                if word2 == 'PAD' or word2 not in self.wvm:
                    batch_embs2[i].append(self.unk)
                else:
                    batch_embs2[i].append(self.wvm[word2])
        return [dy.inputTensor(emb) for emb in batch_embs1], [dy.inputTensor(emb) for emb in batch_embs2]

    def enable_dropout(self):
        self.fwdRNN.set_dropout(0.3)
        self.bwdRNN.set_dropout(0.3)
        self.fwdRNN2.set_dropout(0.3)
        self.bwdRNN2.set_dropout(0.3)

    def disable_dropout(self):
        self.fwdRNN.disable_dropout()
        self.bwdRNN.disable_dropout()
        self.fwdRNN2.disable_dropout()
        self.bwdRNN2.disable_dropout()

    def initialize_paramerets(self):
        # apply dropout
        if self.eval:
            self.disable_dropout()
        else:
            self.enable_dropout()

        # initialize the RNNs
        self.f_init = self.fwdRNN.initial_state()
        self.b_init = self.bwdRNN.initial_state()
        self.f2_init = self.fwdRNN2.initial_state()
        self.b2_init = self.bwdRNN2.initial_state()

    def build_tagging_graph(self, batch):
        self.initialize_paramerets()
        # get the word vectors.
        batch_embs1, batch_embs2 = self.word_rep(batch)

        # feed word vectors into biLSTM
        fw_exps1 = self.f_init.transduce(batch_embs1)
        bw_exps1 = self.b_init.transduce(reversed(batch_embs1))
        fw_exps2 = self.f_init.transduce(batch_embs2)
        bw_exps2 = self.b_init.transduce(reversed(batch_embs2))
    
        # biLSTM states
        bi_exps1 = [dy.concatenate([f,b]) for f,b in zip(fw_exps1, reversed(bw_exps1))]
        bi_exps2 = [dy.concatenate([f,b]) for f,b in zip(fw_exps2, reversed(bw_exps2))]

        # 2nd biLSTM
        fw_exps1 = self.f2_init.transduce(bi_exps1)
        bw_exps1 = self.b2_init.transduce(reversed(bi_exps1))
        fw_exps2 = self.f2_init.transduce(bi_exps2)
        bw_exps2 = self.b2_init.transduce(reversed(bi_exps2))
    
        # biLSTM states
        bi_exps1 = dy.concatenate([dy.concatenate([f,b]) for f,b in zip(fw_exps1, reversed(bw_exps1))], d=1)
        bi_exps2 = dy.concatenate([dy.concatenate([f,b]) for f,b in zip(fw_exps2, reversed(bw_exps2))], d=1)
        aT1 = self.meta.activation(self.aw * bi_exps1 + self.ab)
        aT2 = self.meta.activation(self.aw * bi_exps2 + self.ab)
        alpha1 = self.av * aT1
        alpha2 = self.av * aT2
        attn1 = dy.softmax(alpha1, 1)
        attn2 = dy.softmax(alpha2, 1)
        weighted_sum1 = dy.reshape(bi_exps1 * dy.transpose(attn1), (self.meta.lstm_word_dim*2, ))
        weighted_sum2 = dy.reshape(bi_exps2 * dy.transpose(attn2), (self.meta.lstm_word_dim*2, ))
        if not self.eval:
            weighted_sum1 = dy.dropout(weighted_sum1, 0.3)
            weighted_sum2 = dy.dropout(weighted_sum2, 0.3)
        xh1 = self.meta.activation(self.w1 * weighted_sum1 + self.b1)
        xh2 = self.meta.activation(self.w1 * weighted_sum2 + self.b1)
        xo1 = self.vt * xh1
        xo2 = self.vt * xh2
        return xo1, xo2

    def sent_loss(self, samples):
        self.eval = False
        vecs1, vecs2 = self.build_tagging_graph(samples) 
        for i in range(len(samples)):
            f1 = dy.pick_batch_elem(vecs1, i)
            f2 = dy.pick_batch_elem(vecs2, i)
            self.loss.append(dy.emax([f1-f2+dy.scalarInput(1.0), dy.scalarInput(0.0)]))

    def tag_sent(self, samples):
        dy.renew_cg()
        self.eval = True
        num_true, num_false = 0, 0
        vecs1, vecs2 = self.build_tagging_graph(samples)
        vecs1 = vecs1.value()
        vecs2 = vecs2.value()
        if not isinstance(vecs1, list):
            vecs1 = [vecs1]
            vecs2 = [vecs2]
        for i in range(len(samples)):
            if vecs1[i] <= vecs2[i]:
                num_true += 1
            else:
                num_false += 1
        return num_true, num_false

def make_batches(args, samples):
    data = {}
    bdata = []
    for sent in samples:
        bucket = len(sent[0])
        data.setdefault(bucket, [])
        data[bucket].append(sent)
    data = data.values()
    for bucket in data:
        n = max(1, int(len(bucket) / args.batch_size))
        bdata += [bucket[i::int(n)] for i in range(n)]
    return bdata

def read_train_test_dev(args):
    dev_X = []
    test_X = []
    train_X = []
    dev_files = set(open(args.dev_files).read().split())
    test_files = set(open(args.test_files).read().split())
    with io.open(args.data_file, encoding='utf-8') as fp:
        for i,line in enumerate(fp):
            title, revigion_group, src, tgt = line.strip().split('\t')
            src = src.split()
            tgt = tgt.split()
            if len(src) > 150 or len(tgt) > 150:
                continue
            src = ['<'] + src + ['>']
            tgt = ['<'] + tgt + ['>']
            if len(src) > len(tgt):
                tgt = (tgt+['PAD']*len(src))[:len(src)]
            elif len(src) < len(tgt):
                src = (src+['PAD']*len(tgt))[:len(tgt)]
            if title in test_files:
                test_X.append((src, tgt))
            elif title in dev_files:
                dev_X.append((src, tgt))
            else:
                train_X.append((src, tgt))
    return make_batches(args, train_X), make_batches(args, dev_X), make_batches(args, test_X)

def eval(tagger, dev, ofp=None):
    tagged = []
    good, bad = 0, 0
    for batch in dev:
        true, false = tagger.tag_sent(batch)
        good += true
        bad += false
    print(good/(good+bad))
    sys.stdout.flush()
    return good/(good+bad)

def train_tagger(args, tagger, train, dev):
    pr_acc = 0.0
    n_samples = len(train)
    num_tagged, cum_loss = 0, 0
    status_step = int(5000/args.batch_size) + 2
    print(len(train))
    print(len(dev))
    sys.stdout.flush()
    for ITER in range(args.iter):
        random.shuffle(train)
        tagger.eval = False
        tagger.loss = []
        for i,batch in enumerate(train, 1):
            dy.renew_cg()
            if i % status_step == 0 or i == n_samples:   # print status
                tagger.trainer.status()
                print(cum_loss / num_tagged)
                sys.stdout.flush()
                cum_loss, num_tagged = 0, 0
            tagger.sent_loss(batch)
            num_tagged += args.batch_size #NOTE batch size
            batch_loss = dy.sum_batches(dy.esum(tagger.loss))
            cum_loss += batch_loss.scalar_value()
            batch_loss.backward()
            tagger.trainer.update()
            tagger.loss = []
            dy.renew_cg()
        acc = eval(tagger, dev)
        if acc > pr_acc:
            pr_acc = acc
            print('Save Point:: %d' %ITER)
            if args.save_model:
                tagger.model.save('%s.dy' %args.save_model)
        sys.stdout.flush()
        print("epoch %r finished" % ITER)
        sys.stdout.flush()


def set_labels(args, data):
    trainers = {
        'simsgd'   : dy.SimpleSGDTrainer,
        'cysgd'    : dy.CyclicalSGDTrainer,
        'momsgd'   : dy.MomentumSGDTrainer,
        'adam'     : dy.AdamTrainer,
        'adagrad'  : dy.AdagradTrainer,
        'adadelta' : dy.AdadeltaTrainer,
        'amsgrad'  : dy.AmsgradTrainer
        }
    act_fn = {
        'sigmoid' : dy.logistic,
        'tanh'    : dy.tanh,
        'relu'    : dy.rectify,
        }
    meta = Meta()
    meta.trainer = trainers[args.trainer]
    meta.activation = act_fn[args.act_fn]
    return meta

def load_data(args):
    wvm = None
    train, dev, test = read_train_test_dev(args)
    meta = set_labels(args, train)
    if args.save_model:
        pickle.dump(meta, open('%s.meta' %args.save_model, 'wb'))
    return train, dev, meta

def main():
    parser = ArgumentParser(description="LSTM Ranking Model")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('--dynet-gpu')
    parser.add_argument('--dynet-mem')
    parser.add_argument('--dynet-devices')
    parser.add_argument('--dynet-autobatch')
    parser.add_argument('--dynet-seed', dest='seed', type=int, default='127')
    parser.add_argument('--data_file', help='wikiHow data file')
    parser.add_argument('--test_files')
    parser.add_argument('--dev_files')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--pre_word_vec', help='Pretrained word2vec Embeddings')
    parser.add_argument('--bin_vec', type=int, help='1 if binary embedding file else 0')
    parser.add_argument('--elimit', type=int, default=None, help='load only top-n pretrained word vectors (default=all vectors)')
    parser.add_argument('--trainer', default='amsgrad', help='Trainer [momsgd|adam|adadelta|adagrad|amsgrad]')
    parser.add_argument('--activation', dest='act_fn', default='tanh', help='Activation function [tanh|rectify|logistic]')
    parser.add_argument('--iter', type=int, default=25, help='No. of Epochs')
    group.add_argument('--save-model', dest='save_model', help='Specify path to save model')
    group.add_argument('--load-model', dest='load_model', help='Load Pretrained Model')
    parser.add_argument('--output-file', dest='outfile', default='/tmp/out.txt', help='Output File')
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    wvm = gensim.models.KeyedVectors.load_word2vec_format(args.pre_word_vec, binary=args.bin_vec, limit=args.elimit)

    if args.load_model:
        sys.stdout.write('Loading Models ...\n')
        tagger = Tagger(model=args.load_model, wvm=wvm)
        train, dev, test = read_train_test_dev(args)
        sys.stdout.write('Done!\n')
        eval(tagger, test)
    else:
        # load data
        train, dev, meta = load_data(args)
        meta.w_dim_e = wvm.syn0.shape[1]
        # initialize parser
        tagger = Tagger(meta=meta, wvm=wvm)
        train_tagger(args, tagger, train, dev)

if __name__ == '__main__':
    main()
