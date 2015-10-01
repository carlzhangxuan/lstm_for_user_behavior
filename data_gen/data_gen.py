#coding:utf8

import pymc as pm
import numpy as np
from sklearn.cluster import KMeans


def data_gen(samples_n=10, tau_start=75, tau_end=100, gamma=0.1):
    alpha = 1./gamma
    for x in xrange(samples_n):
        tau = pm.rdiscrete_uniform(tau_start, tau_end)
        #lam = pm.rexponential(alpha)
        lam = alpha
        yield pm.rpoisson(lam, tau)
        #yield (tau, lam)

def data_gen_for_rnn(samples_n=1, tau_start=75, tau_end=100, gamma=0.01, var=5):
    alpha = 1./gamma
    lam = alpha
    for i in xrange(samples_n):
        con = []
        tau = pm.rdiscrete_uniform(tau_start, tau_end)
        for j in xrange(tau):
            if j == 0:
                val = round(pm.rnormal(lam, var), 2)
                con.append(val)
            elif j == 1:
                val = con[0] + pm.rnormal(0, var)
                val = round(val, 2)
                con.append(val)

            else:
                #n = len(con)
                #lam_n = float(np.array(con).sum())/n
                val = 0.7*con[-1] + 0.3*con[-2] + pm.rnormal(0, var)
                val = round(val, 2)
                con.append(val)
                #print val, lam_n
        yield con


def get_fft(seq):
    return np.fft.rfft(seq)

def test_fft():
    my_matrix = []
    test_data_1 = data_gen()
    test_data_2 = data_gen(gamma=0.15)
    for x in test_data_1:
        #print x
        print get_fft(x)[:4]
        my_matrix.append(get_fft(x)[:4])
    print '---------------------'
    for x in test_data_2:
        #print x
        print get_fft(x)[:4]
        my_matrix.append(get_fft(x)[:4])
    
    X = np.array(my_matrix)
    #print X
    model = KMeans(5)
    model.fit(X)
    res = model.predict(X)
    return res

def data_rnn():
    data_t = data_gen_for_rnn(tau_start=750, tau_end=1000)
    for rec in data_t:
        for x in xrange(len(rec)-1):
            print rec[x], rec[x+1]

if __name__ == '__main__':
    #print test_fft()
    data_rnn()
