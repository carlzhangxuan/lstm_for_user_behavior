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

def get_data(fn='rnn_1'):
    with open(fn) as f:
        for line in f:
            yield np.asarray(line.strip('\n').split(' '))
def refine_data():
    pass

def main_loop():
    print 'loading data...'
    

class TestMainLoop(unittest.TestCase):
    
    def test_main_loop(self):
        print 'testing main loop...'
        main_loop()

if __name__ == '__main__':
    print 'training lstm with unit test...'
    unittest.main()
