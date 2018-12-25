# encoding: utf-8

"""
Creating an RNN summarizer...

TODO check out https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/summarization_tutorial.ipynb
"""

import collections
from nltk.tokenize import word_tokenize, sent_tokenize
import tensorflow as tf

text = 'long ago , the mice had a general council to consider what measures they could take to outwit their ' \
       'common enemy , the cat . some said this , and some said that but at last a young mouse got up and said' \
       ' he had a proposal to make , which he thought would meet the case . you will all agree , said he , that' \
       ' our chief danger consists in the sly and treacherous manner in which the enemy approaches us . now , if ' \
       'we could receive some signal of her approach , we could easily escape from her . i venture , therefore , to' \
       ' propose that a small bell be procured , and attached by a ribbon round the neck of the cat . by this means ' \
       'we should always know when she was about , and could easily retire while she was in the neighbourhood . this ' \
       'proposal met with general applause , until an old mouse got up and said that is all very well , but who is ' \
       'to bell the cat ? the mice looked at one another and nobody spoke . then the old mouse said it is easy to ' \
       'propose impossible remedies .'


text_words = word_tokenize(text)


def build_datasets(words):
    """ the names says it.."""
    count = collections.Counter(words).most_common()
    dictionary = dict([(word_count[0], i_index) for i_index, word_count in enumerate(count)])
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary


def rnn(x, weights, biases):
    """ input rnn... lol """
    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    # 1-layer LSTM with n_hidden units.
    rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']