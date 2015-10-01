#coding:utf8

import numpy as np
import theano
import os
import sys
import cPickle as pickle
import unittest

imdb_data = '/home/disk3/zhangxuan/DL_Pipeline/data/imdb.pkl'
imdb_dict_data = '/home/disk3/zhangxuan/DL_Pipeline/data/imdb.dict.pkl'

def doc2bow(docs):
    worddict = {}
    word_id = 0
    for line in docs:
        for word in line:
            if word not in worddict:
                worddict[word] = word_id
    new_docs = []
    for line in docs:
        new_docs.append(worddicr[word] for word in line)
    return (new_docs, worddict)

def docing(docs):
    return np.asarray(zip(docs[0], docs[1]))

def load_data(samples, sample_dict, max_sample_len=None, max_word_id=None, seed=None, valid_portion=0.2, test_portion=0.1, sorted_by_len=True):
    np.random.seed(seed)
        
    with open(samples) as f_data:
        ori_samples = pickle.load(f_data)

    samples_with_labels = docing(ori_samples)
    
    if max_sample_len != None:
        assert type(max_sample_len) is int
        samples_with_labels = filter(lambda (x, y):(len(x)) <= max_sample_len, samples_with_labels)
    
    if max_word_id != None:
        assert type(max_word_id) is int
        samples_with_labels = map(lambda (X,y):([x if x<= max_word_id else 1 for x in X], y), samples_with_labels)

    samples_len = len(samples_with_labels)
    samples_idx = np.random.permutation(samples_len)
    valid_len = int(np.ceil(samples_len*valid_portion))
    test_len = int(np.ceil(samples_len*test_portion))
    train_len = samples_len - valid_len - test_len
    train_data = np.array(samples_with_labels[:train_len])
    valid_data = np.array(samples_with_labels[train_len:train_len+valid_len])
    test_data = np.array(samples_with_labels[train_len+valid_len:])
    
    if sorted_by_len == True:
        train_data = np.array(sorted(train_data, key=lambda x:len(x[0])))
        valid_data = np.array(sorted(valid_data, key=lambda x:len(x[0])))
        test_data = np.array(sorted(test_data, key=lambda x:len(x[0])))
    
    return train_data, valid_data, test_data

def prepare_data(data_set, max_sample_len=None):

    if max_sample_len != None:
        assert type(max_sample_len) is int
        data_set = np.array(filter(lambda (x, y):(len(x)) <= max_sample_len, data_set))

    samples_len = len(data_set)
    max_sample_len = np.array([len(s) for s in data_set[:, 0]]).max()
    x = np.zeros((max_sample_len, samples_len)).astype('int64')
    x_mask = np.zeros((max_sample_len, samples_len)).astype(theano.config.floatX)
    for idx, s in enumerate(data_set[:, 0]):
        x[:len(s), idx] = s
        x_mask[:len(s), idx] = 1
    
    return x, x_mask, data_set[:,1] 
            
class TestLoadData(unittest.TestCase):
    
    def test_load_data(self):
        imdb_data = '/home/disk3/zhangxuan/DL_Pipeline/data/imdb.pkl'
        imdb_dict_data = '/home/disk3/zhangxuan/DL_Pipeline/data/imdb.dict.pkl'
        print 'testing load_data ... '
        train_set, valid_set, test_set = load_data(samples=imdb_data, sample_dict=imdb_dict_data, max_sample_len=30, max_word_id = 10000, seed=123)
        prepared_data = prepare_data(train_set)
        print prepared_data

if __name__ == '__main__':
    unittest.main()
