# Skip gram

# --------------------------------------------------------------------------- #
# Load libraries
# --------------------------------------------------------------------------- #

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import tensorflow as tf
import math

import nltk

import re
import itertools
import collections

import numpy as np
from six.moves import xrange


# --------------------------------------------------------------------------- #
# Used functions
# --------------------------------------------------------------------------- #

class WordEmbedding(object):
    """
    Super class of starting your word embedding...
    Can load the data.. prepare it..
    The subclasses can learn the word embedding

    There is a change to this class going on.. since data preparation can be quite.. different from time to time,
    but running the model itself will always be the same.
    Hence, we are going to organize the class in such a way that you need to initialize it with the labeled data (in
    integer format), and provide a vocabulary.
    """

    def __init__(self, input_label, vocab_dict, batch_size, num_sampled=10, embedding_size=300):
        # self.raw_data = raw_data
        # self.data = raw_data
        # self.num_skips = num_skips  # used for prepping data according to tensorflow
        # self.skip_window = skip_window  # used for prepping data according to tensorflow
        self.batch_size = batch_size
        self.num_sampled = num_sampled  # used in the nce-loss
        self.input_label = input_label
        self.n_steps = len(input_label) // batch_size
        self.embedding_size = embedding_size
        self.vocab_dict = vocab_dict
        self.vocabulary_size = len(vocab_dict)
        self.embedding = []
        self.data_index = 0

    def gen_string_data_set(self, n_words=50000):
        """
        This function is used to obtain a dictionary of words based on one big ass string.
        This function will split the big ass string into sentences, and then into words

        Some sample books
        https://storage.googleapis.com/hebban-website-eu.appspot.com/files/pdf/00000186/542cddfde273f7.38020880.pdf
        https://www.u-cursos.cl/usuario/3ad9e52ee57c6df8595d35790bad7c32/mi_blog/r/[Harry_Mulisch_De_Ontdekking_van_de_Hemel_BookFi_(1).pdf

        :param n_words: return the n_words amount of top words -> not working ATM
        """

        data = re.sub("\\n", " ", self.raw_data)
        data = re.sub("‘|’|“|”|«|»", "", data)
        # sentence = nltk.tokenize.sent_tokenize(data, language='dutch')  # Dude what is the use of this...
        # sentence = ' '.join(sentence)  # Dude what is the use of this...
        words = nltk.tokenize.word_tokenize(data, language='dutch')
        unique_words = np.unique(words)
        words_dict_reverse = dict(enumerate(unique_words))

        self.vocab_dict = words_dict_reverse

        words_dict = dict((v, k) for k, v in words_dict_reverse.items())
        words_int = [words_dict.get(x) for x in words]
        # TODO: with this I could subset the amount of words in the vocab.
        words_count = collections.Counter(words).most_common(n_words - 1)

        self.input_label = words_int
        self.vocabulary_size = len(words_dict)

        return self.data

    def get_batch(self):
        """
         When a full set of input/labels is given. We can retrieve it with this one.
        """
        assert len(self.input_label) > 0
        if self.data_index >= len(self.input_label):
            self.data_index = 0
        start_index = self.data_index
        end_index = self.data_index + self.batch_size
        self.data_index = end_index
        return self.input_label[start_index:end_index]

    def generate_batch(self):
        """
        Returns a tuple of batch and label data.. however it also uses a global data_index, something I dont like
        Further, given a dataset.. this function then makes sure that everything is sorted out for skipgram
        """
        assert self.batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.skip_window

        batch = np.ndarray(shape=self.batch_size, dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)

        if self.data_index + span > len(self.data):
            self.data_index = 0
        buffer.extend(self.data[self.data_index:self.data_index + span])
        self.data_index += span
        for i in range(self.batch_size // self.num_skips):
            context_words = [w for w in range(span) if w != self.skip_window]
            np.random.shuffle(context_words)
            words_to_use = collections.deque(context_words)
            for j in range(self.num_skips):
                batch[i * self.num_skips + j] = buffer[self.skip_window]
                context_word = words_to_use.pop()
                labels[i * self.num_skips + j, 0] = buffer[context_word]
            if self.data_index == len(self.data):
                buffer[:] = self.data[:span]
                self.data_index = span
            else:
                buffer.append(self.data[self.data_index])
                self.data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
        return batch, labels

    def _skip_gram_character(self, learning_rate):
        """
        Stuff...
        :param learning_rate:
        :return:
        """
        return NotImplementedError

    def _skip_gram(self, learning_rate):
        """
        Stuff
        :param learning_rate:
        :return:
        """
        return NotImplementedError

    def run(self, learning_rate):
        """
        Run something..

        :param learning_rate:
        :return:
        """
        return NotImplementedError


class CBOW(WordEmbedding):
    """
    Dingen

    """
    def __init__(self, input_label, vocab_dict, batch_size, num_sampled=10, embedding_size=300):
        super(CBOW).__init__(input_label, vocab_dict, batch_size, num_sampled=num_sampled,
                             embedding_size=embedding_size)
        # This one still needs to be tested

    def generate_batch(self):
        """
        Batch generator for CBOW (Continuous Bag of Words).
        batch should be a shape of (batch_size, num_skips)
        Parameters
        ----------
        data: list of index of words
        batch_size: number of words in each mini-batch
        num_skips: number of surrounding words on both direction (2: one word ahead and one word following)
        skip_window: number of words at both ends of a sentence to skip (1: skip the first and last word of a sentence)
        """
        assert self.batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.skip_window

        batch = np.ndarray(shape=self.batch_size, dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)

        # collect the first window of words
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        # move the sliding window
        for i in range(self.batch_size):
            mask = [1] * span
            mask[self.skip_window] = 0
            batch[i, :] = list(itertools.compress(buffer, mask))  # all surrounding words
            labels[i, 0] = buffer[self.skip_window]  # the word at the center
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        return batch, labels


class CharacterSkipGram(WordEmbedding):
    """
    I think we are also still testing this one..
    """
    def __init__(self, input_label, vocab_dict, batch_size, ngram_batch_size, num_sampled=10, embedding_size=300):
        super(CharacterSkipGram).__init__(input_label, vocab_dict, batch_size, num_sampled=num_sampled,
                         embedding_size=embedding_size)
        self.ngram_batch_size = ngram_batch_size
        self.ngram_vocab_dict = dict()
        self.ngram_vocabulary_size = len(self.ngram_vocab_dict)

    def _skip_gram_character(self, learning_rate):
        """
        This one is already adapted to character skipgram...
        :param learning_rate:
        :return:
        """
        graph_skipgram_char = tf.Graph()

        with graph_skipgram_char.as_default():
            batch_input = tf.placeholder(tf.int32, shape=[self.batch_size, ], name='batchinput')
            ngram_batch_input = tf.placeholder(tf.int32, shape=[self.ngram_batch_size, ], name='ngraminput')
            batch_label = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='batchlabel')

            with tf.variable_scope('char_skipgram'):
                init = tf.global_variables_initializer()
                embedding = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0),
                                        name='embedding')
                ngram_embedding = tf.Variable(tf.random_uniform([self.ngram_vocabulary_size, self.embedding_size],
                                                                -1.0, 1.0), name='ngram_embedding')
                batch_embedding = tf.nn.embedding_lookup(embedding, batch_input, name='batch_embed')
                batch_ngram_embedding = tf.nn.embedding_lookup(ngram_embedding, ngram_batch_input,
                                                               name='batch_ngram_embed')

                ngram_sum = tf.reduce_sum(batch_ngram_embedding, axis=0, name='reduce_sum')
                character_embedding = tf.add(ngram_sum, batch_embedding, name='new_embed')

                weight = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                                         stddev=1.0/math.sqrt(self.embedding_size)), name='weights')
                bias = tf.Variable(tf.zeros([self.vocabulary_size]), name='bias')
                loss = tf.reduce_mean(tf.nn.nce_loss(weights=weight,
                                                     biases=bias,
                                                     labels=batch_label,
                                                     inputs=character_embedding,
                                                     num_sampled=self.num_sampled,
                                                     num_classes=self.vocabulary_size), name='loss')
                norm = tf.sqrt(tf.reduce_mean(tf.square(embedding), 1, keep_dims=True))
                normalized_embedding = tf.divide(embedding, norm, name='norm_embed')
                optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, name='optimizer')
                # optimizer = tf.train.AdamOptimizer(0.8).minimize(loss, name='optimizer')

                # val_embedding = tf.nn.embedding_lookup(normalized_embedding, val_dataset)
                # similarity = tf.matmul(val_embedding,normalized_embedding,transpose_b = True)
        return graph_skipgram_char

    def run(self, learning_rate):
        """
        # TODO this should be changed... now this is the same as the stadnard skipgram model
        :return:
        """
        skip_gram_graph = self._skip_gram_character(learning_rate=learning_rate)
        batch_input = skip_gram_graph.get_operation_by_name('batchinput')
        batch_label = skip_gram_graph.get_operation_by_name('batchlabel')
        normalized_embedding = skip_gram_graph.get_operation_by_name('norm_embed').values()
        loss = skip_gram_graph.get_operation_by_name('loss')
        optimizer = skip_gram_graph.get_operation_by_name('optimizer')

        loss_value_list = []
        average_loss = 0.0

        with tf.Session(graph=skip_gram_graph) as sess:
            tf.global_variables_initializer().run()

            for step in xrange(self.n_steps):
                if step % 1000 == 0:
                    print(step)
                input_data, label = self.generate_batch()
                feed_dict = {batch_input: input_data, batch_label: label}
                loss_val, _ = sess.run([loss, optimizer], feed_dict)
                average_loss += loss_val
                loss_value_list.append(loss_val)

            final_embedding = normalized_embedding.eval()
            self.embedding = final_embedding
            return loss_value_list, final_embedding


class SkipGram(WordEmbedding):
    """
    If raw_data (input) is one long ass string, THEN this can be transformed to a proper dataset with the function
    * gen_string_data_set

    If raw_data (input) is a dataframe/numpy array, THEN this can be transformed to a proper dataset with the function
    * gen_oms_data_set

    """
    def __init__(self, input_label, vocab_dict, batch_size, num_sampled=10, embedding_size=300):
        super(SkipGram).__init__(input_label, vocab_dict, batch_size, num_sampled=num_sampled,
                                 embedding_size=embedding_size)

    def _skip_gram(self, learning_rate):
        g1 = tf.Graph()

        with g1.as_default():
            batch_input = tf.placeholder(tf.int32, shape=[self.batch_size, ], name='batchinput')
            batch_label = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='batchlabel')

            with tf.variable_scope('skipgram'):
                init = tf.global_variables_initializer()
                embedding = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0),
                                        name='embedding')
                batch_embedding = tf.nn.embedding_lookup(embedding, batch_input, name='batch_embed')
                weight = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                                         stddev=1.0/math.sqrt(self.embedding_size)), name='weights')
                bias = tf.Variable(tf.zeros([self.vocabulary_size]), name='bias')
                loss = tf.reduce_mean(tf.nn.nce_loss(weights=weight,
                                                     biases=bias,
                                                     labels=batch_label,
                                                     inputs=batch_embedding,
                                                     num_sampled=self.num_sampled,
                                                     num_classes=self.vocabulary_size), name='loss')
                norm = tf.sqrt(tf.reduce_mean(tf.square(embedding), 1, keep_dims=True))
                normalized_embedding = tf.divide(embedding, norm, name='norm_embed')
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, name='optimizer')
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, name='optimizer')

                # val_embedding = tf.nn.embedding_lookup(normalized_embedding, val_dataset)
                # similarity = tf.matmul(val_embedding,normalized_embedding,transpose_b = True)
        return g1

    def run(self, learning_rate):
        """
        Here we can run the model.. I guess...

        :param learning_rate:
        :return:
        """
        skip_gram_graph = self._skip_gram(learning_rate=learning_rate)
        batch_input = skip_gram_graph.get_operation_by_name('batchinput').values()
        batch_label = skip_gram_graph.get_operation_by_name('batchlabel').values()
        normalized_embedding = skip_gram_graph.get_operation_by_name('skipgram/norm_embed').values()
        loss = skip_gram_graph.get_operation_by_name('skipgram/loss').values()
        optimizer = skip_gram_graph.get_operation_by_name('skipgram/optimizer')

        loss_value_list = []
        average_loss = 0.0
        loss_val = [0]

        with tf.Session(graph=skip_gram_graph) as sess:
            tf.global_variables_initializer().run()

            for step in xrange(self.n_steps):
                if step % (self.n_steps // 100) == 0:
                    loss_value_list.append(loss_val)
                    print(step, loss_val[0])
                # input, label = self.generate_batch()
                input_label = self.get_batch()
                input_data, label = list(zip(*input_label))
                label = np.reshape(label, (self.batch_size, 1))
                input_data = np.array(input_data)

                feed_dict = {batch_input: input_data, batch_label: label}
                loss_val, _ = sess.run([loss, optimizer], feed_dict=feed_dict)

                average_loss += loss_val[0]
                # loss_value_list.append(loss_val)

            final_embedding = sess.run(normalized_embedding, feed_dict=feed_dict)
            self.embedding = final_embedding
            return loss_value_list, final_embedding
