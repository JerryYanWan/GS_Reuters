"""
    Theano implementation of Convolutional Neural Network
    One Convolutional Layer, one maxpooling layer, one feedforward layer.
    
    Written by YanWan.
    25-8-2016, HKUST-HLTC
"""

#__author__ == 'YanWan'

import theano
import theano.tensor as T
import numpy as np
from theano.compat.python2x import OrderedDict

class CNN(object):
    def __init__(self, x_input, tag_lookup_tbl, \
                 word_vec_dim, conv_size, hidden_dim, embed_dim):
        self.conv_layer = theano.shared( name = 'convolution_layer',
                              value = 0.01 * np.sqrt(6) / np.sqrt(word_vec_dim * conv_size + hidden_dim) * np.random.uniform(-1., 1., (word_vec_dim * conv_size, hidden_dim)).astype(theano.config.floatX), borrow=True )
        self.hidden_layer = theano.shared( name = 'hidden_layer',
                              value = 0.01 * np.sqrt(6) / np.sqrt(hidden_dim + embed_dim) * np.random.uniform(-1., 1., (hidden_dim, embed_dim)).astype(theano.config.floatX), borrow=True )
        self.tag_hidden_layer = theano.shared( name = 'tag_hidden_layer',
                              value = 0.01 * np.sqrt(6) / np.sqrt(word_vec_dim + embed_dim) * np.random.uniform(-1., 1., (word_vec_dim, embed_dim)).astype(theano.config.floatX), borrow=True )

        self.h_conv_layer = theano.shared( name = 'h_conv', 
                              value = np.zeros(hidden_dim, dtype=theano.config.floatX), borrow = True )
        self.h_hidden_layer = theano.shared( name = 'h_hidden', 
                              value = np.zeros(embed_dim, dtype=theano.config.floatX), borrow = True )
        self.h_tag_hidden_layer = theano.shared( name = 'h_tag_hidden', 
                              value = np.zeros(embed_dim, dtype=theano.config.floatX), borrow = True )

        self.params = [ self.conv_layer, self.h_conv_layer, \
                        self.hidden_layer, self.h_hidden_layer, \
                        self.tag_hidden_layer, self.h_tag_hidden_layer ]

        self.input = x_input
        self.tag_lookup_tbl = tag_lookup_tbl
        def convolution(idx, matr, length):
            cur_conv = matr[idx:idx+length].ravel()
            return T.tanh(T.dot(cur_conv, self.conv_layer) + self.h_conv_layer)

        slice_conv_index = T.arange(0, self.input.shape[0] - conv_size + 1, conv_size)
        self.x_conv, updates = theano.scan( fn = convolution,
                                            sequences = [slice_conv_index],
                                            non_sequences = [self.input, conv_size] )
        self.hidden = T.tanh(T.max(self.x_conv, 0))
        self.embed  = T.dot(self.hidden, self.hidden_layer) + self.h_hidden_layer
        self.tag_lookup = T.dot(self.tag_lookup_tbl, self.tag_hidden_layer) + self.h_tag_hidden_layer
        index_sequence = T.arange(0, self.tag_lookup.shape[0], 1)
        def cosine(idx, matr, vctr):
            v1 = matr[idx, :]
            return (1+T.sum(v1*vctr) / T.sqrt(T.sum(v1**2)*T.sum(vctr**2)))/2
        similarity, updates = theano.scan( fn = cosine,
                                           sequences = [index_sequence],
                                           non_sequences = [self.tag_lookup, self.embed] )
        self.score = similarity

    def diff(self, tag):
        copy = self.score
        benchmark = T.min(copy[tag])
        return copy - benchmark
    def loss(self, tag):
        return - T.mean(T.log(self.score[tag]))
    def rank(self):
        return self.score
    def lookup(self):
        return self.tag_lookup
          
