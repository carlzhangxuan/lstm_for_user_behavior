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

def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

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

def lstm(tparams, state_below, model_options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n+1)*dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[prefix + '_U'])
        preact += x_
        
        i = tensor.nnet.sigmoid(_slice(preact, 0, model_options['word_embeding_dimension']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, model_options['word_embeding_dimension']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, model_options['word_embeding_dimension']))
        c = tensor.tanh(_slice(preact, 3, model_options['word_embeding_dimension'])) 
        
        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
    
        return h, c
    
    state_below = (tensor.dot(state_below, tparams[prefix + '_W']) + tparams[prefix + '_b'])
    dim_proj = model_options['word_embeding_dimension']
    rval, updates = theano.scan(_step, sequences=[mask, state_below], outputs_info=[tensor.alloc(numpy_floatX(0.), n_samples, dim_proj), tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)], name= prefix+'_layers', n_steps=nsteps)
    return rval[0]

def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise, (state_before * trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype)),state_before * 0.5)
    return proj

def build_model(tparams, model_options):
    trng = RandomStreams(SEED)
    use_noise = theano.theano.shared(numpy_floatX(0.))
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')
    n_timesteps = x.shape[0]
    n_samples = x.shape[1] 
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, model_options['word_embeding_dimension']])
    #proj = lstm(tparams, emb, model_options, prefix='lstm', mask=mask)
    #proj = (proj * mask[:, :, None]).sum(axis=0)
    #proj = proj / mask.sum(axis=0)[:, None]
    
    #if model_options['use_dropout']:
    #    proj = dropout_layer(proj, use_noise, trng)

    #pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])
    #f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    #f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    #off = 1e-8
    #if pred.dtype == 'float16':
    #    off = 1e-6
    
    #cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    #tmp
    f_pred_prob, f_pred, cost = 0, 0, 0

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost

    
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
    (use_noise, x, mask, y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

class TestMainLoop(unittest.TestCase):
    
    def test_main_loop(self):
        print 'testing main loop...'
        main_loop()

if __name__ == '__main__':
    print 'training lstm with unit test...'
    unittest.main()
