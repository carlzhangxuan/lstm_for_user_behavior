#coding:utf8

import sys
sys.path.append('/home/disk3/zhangxuan/DL_Pipeline')
import time
import unittest
import cPickle as pkl
import numpy as np
import theano
import theano.tensor as tensor
import traceback
import logging

from collections import OrderedDict
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from optimizer import optimizer

SEED = 312

def get_data(fn='data/rnn.txt'):
    with open(fn) as f:
        for line in f:
            yield line.strip('\n').split(' ')

def refine_data(oridata_gen=get_data):
    con = []
    for line in oridata_gen():
        con.append((float(line[0]), 0 if float(line[1]) >=99 else 1))
    return np.array(con)

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)

def init_paras(model_options):
    params = OrderedDict()
    randn = np.random.rand(model_options['max_word_id'], model_options['word_embeding_dimension'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = init_lstm_paras(model_options, params, prefix='lstm')
    params['U'] = 0.01 * np.random.randn(model_options['word_embeding_dimension'], model_options['ydim']).astype(config.floatX)
    params['b'] = np.zeros((model_options['ydim'],)).astype(config.floatX)
    return params

def init_lstm_paras(options, params, prefix='lstm'):
    W = np.concatenate([ortho_weight(options['word_embeding_dimension']), ortho_weight(options['word_embeding_dimension']), ortho_weight(options['word_embeding_dimension']), ortho_weight(options['word_embeding_dimension'])], axis=1)
    params[prefix + '_W'] = W
    U = np.concatenate([ortho_weight(options['word_embeding_dimension']), ortho_weight(options['word_embeding_dimension']), ortho_weight(options['word_embeding_dimension']), ortho_weight(options['word_embeding_dimension'])], axis=1)
    params[prefix + '_U'] = U
    b = np.zeros((4 * options['word_embeding_dimension'],))
    params[prefix + '_b'] = b.astype(config.floatX)
    return params

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)

    return tparams

def main_loop(samples=refine_data, word_embeding_dimension=128, max_word_id=1, use_dropout=True):
    
    #getting settings
    model_options = locals().copy()  

    #loading data
    print 'loading data...'
    train_set = samples()
    print 'trainingset shape: %srows, %sclos' % train_set.shape
    ydim = np.max(train_set[:,1]) + 1
    model_options['ydim'] = ydim
    print 'model_options: %s' % model_options

    #setting paras
    params = init_paras(model_options)
    tparams = init_tparams(params)

    #building model
    

class TestMainLoop(unittest.TestCase):
    
    def test_main_loop(self):
        print 'testing main loop...'
        main_loop()

if __name__ == '__main__':
    print 'training lstm with unit test...'
    unittest.main()
