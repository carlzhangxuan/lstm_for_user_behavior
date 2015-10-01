import cPickle as pkl
import sys
import time
import numpy as np
import theano
import theano.tensor as tensor
import traceback

from collections import OrderedDict
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from optimizer import optimizer
from data import data_loader

SEED = 123

loader = data_loader.load_data
preparer = data_loader.prepare_data

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

def init_lstm_paras(options, params, prefix='lstm'):
    W = np.concatenate([ortho_weight(options['word_embeding_dimension']), ortho_weight(options['word_embeding_dimension']), ortho_weight(options['word_embeding_dimension']), ortho_weight(options['word_embeding_dimension'])], axis=1)
    params[prefix + '_W'] = W
    U = np.concatenate([ortho_weight(options['word_embeding_dimension']), ortho_weight(options['word_embeding_dimension']), ortho_weight(options['word_embeding_dimension']), ortho_weight(options['word_embeding_dimension'])], axis=1)
    params[prefix + '_U'] = U
    b = np.zeros((4 * options['word_embeding_dimension'],))
    params[prefix + '_b'] = b.astype(config.floatX)

    return params

def init_paras(model_options):
    params = OrderedDict()
    randn = np.random.rand(model_options['max_word_id'], model_options['word_embeding_dimension'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = init_lstm_paras(model_options, params, prefix='lstm')
    params['U'] = 0.01 * np.random.randn(model_options['word_embeding_dimension'], model_options['ydim']).astype(config.floatX)
    params['b'] = np.zeros((model_options['ydim'],)).astype(config.floatX)
    return params
    
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)

    return tparams

def lstm(tparams, state_below, model_options, prefix='lstm', mask =None):
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

def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    n_samples = len(data[:,0])
    probs = np.zeros((n_samples, 2)).astype(config.floatX)
    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data(np.array(zip([data[:,0][t] for t in valid_index], np.array(data[:,1])[valid_index])), max_sample_len=None)

        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs
        n_done += len(valid_index)

        if verbose:
            print '%d/%d samples classified' % (n_done, n_samples)
    return probs

def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data(np.array(zip([data[:,0][t] for t in valid_index], np.array(data[:,1])[valid_index])), max_sample_len=None)
        preds = f_pred(x, mask)
        targets = np.array(data[:,1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])
    return valid_err


def build_model(tparams, model_options):
    trng = RandomStreams(SEED)
    use_noise = theano.shared(numpy_floatX(0.))
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')
    
    n_timesteps = x.shape[0]
    #max_len
    n_samples = x.shape[1]
    #?

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, model_options['word_embeding_dimension']])
    #Wemb->(10000, 128),(10000?one_hot?*sentence_len?)->(10000*sentence_len*dim)
    proj = lstm(tparams, emb, model_options, prefix='lstm', mask=mask)

    proj = (proj * mask[:, :, None]).sum(axis=0)
    proj = proj / mask.sum(axis=0)[:, None]
    if model_options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])
    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6
    
    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()
    return use_noise, x, mask, y, f_pred_prob, f_pred, cost

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32") 
    if shuffle:np.random.shuffle(idx_list)
    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start: minibatch_start + minibatch_size])
        minibatch_start += minibatch_size
    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])
    return zip(range(len(minibatches)), minibatches)

def train_model(samples, sample_dict, max_sample_len=50, max_word_id = 10000, seed=123, word_embeding_dimension=128, use_dropout=True, valid_batch_size=64, batch_size=16,\
                decay_c=0., lrate=0.0001, validFreq=370, saveFreq=1110, dispFreq=10, max_epochs=500, patience=10):
    
    #params init&data load
    model_options = locals().copy()
    train_set, valid_set, test_set = loader(samples, sample_dict, max_sample_len, max_word_id, seed)
    ydim = np.max(train_set[:,1]) + 1
    model_options['ydim'] = ydim
    params = init_paras(model_options)
    tparams = init_tparams(params)
    #build model
    (use_noise, x, mask, y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y], cost, name='f_cost')
    grads = tensor.grad(cost, wrt=tparams.values())
    f_grad = theano.function([x, mask, y], grads, name='f_grad')
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer.adadelta(lr, tparams, grads, x, mask, y, cost)
    print 'Optimization'

    kf_valid = get_minibatches_idx(len(valid_set[:,0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test_set[:, 0]), valid_batch_size)
    print "%d train examples" % len(train_set[:,0])
    print "%d valid examples" % len(valid_set[:,0])
    print "%d test examples" % len(test_set[:,0])
    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

    uidx = 0
    estop = False
    start_time = time.time()

    #train_testing loop
    try:
        for eidx in xrange(max_epochs):
            n_samples = 0
            kf = get_minibatches_idx(len(train_set[:,0]), batch_size)

            for oidx, train_idx in kf:
                uidx += 1
                use_noise.set_value(1.0)

                y = [train_set[:,1][t] for t in train_idx]
                x = [train_set[:,0][t] for t in train_idx]

                x, mask, y =  preparer(np.array(zip(x, y)))
                y = list(y)

                n_samples += x.shape[1]
                cost = f_grad_shared(x, mask, y)
                f_update(lrate)
                
                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                if np.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, preparer, train_set, kf)
                    valid_err = pred_error(f_pred, preparer, valid_set, kf_valid)
                    test_err = pred_error(f_pred, preparer, test_set, kf_test)
                    history_errs.append([valid_err, test_err])

                    if (uidx == 0 or valid_err <= np.array(history_errs)[:,0].min()):
                        best_p = unzip(tparams)
                        bad_counter = 0
                    print ('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err)
                    if (len(history_errs) > patience and valid_err >= np.array(history_errs)[:-patience, 0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break
            
            print 'Seen %d samples' % n_samples
                                    
    except Exception,e:
        print traceback.format_exc()
    
    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train_set[:,0]), batch_size)
    train_err = pred_error(f_pred, preparer, train_set, kf_train_sorted)
    valid_err = pred_error(f_pred, preparer, valid_set, kf_valid)
    test_err = pred_error(f_pred, preparer, test_set, kf_test)

    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
    print 'The code run for %d epochs, with %f sec/epochs' % ((eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' % (end_time - start_time))

    return train_err, valid_err, test_err


if __name__ == '__main__':
    imdb_data = '/home/disk3/zhangxuan/DL_Pipeline/data/imdb.pkl'
    imdb_dict_data = '/home/disk3/zhangxuan/DL_Pipeline/data/imdb.dict.pkl'
    train_model(samples=imdb_data, sample_dict=imdb_dict_data, max_epochs=100)
