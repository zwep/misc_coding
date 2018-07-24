# Apparantly this is now the most up to date turoitral on neural translation machines
# https://github.com/tensorflow/nmt
# Here is some guy making tutorials
# https://github.com/ematvey/tensorflow-seq2seq-tutorials/tree/f767fd66d940d7852e164731cc774de1f6c35437
# ere is ANother one
# https://github.com/llSourcell/seq2seq_model_live/blob/master/2-seq2seq-advanced.ipynb
# Here is some part about why to use raw_rnn instead of dynamic_rnn
# https://hanxiao.github.io/2017/08/16/Why-I-use-raw-rnn-Instead-of-dynamic-rnn-in-Tensorflow-So-Should-You-0/

# more examples
# https://github.com/JayParks/tf-seq2seq
# https://github.com/ilblackdragon/tf_examples/blob/master/seq2seq/seq2seq.py


import nltk
import os
import functools  # Used to get an advanced type of callable object
import numpy as np
import pandas as pd

import tensorflow as tf


"""
Basic info:

We are trying to run some simple seq2seq model here... 

data is obtained via stanford english-vietnamese dict: https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/

model is obtained through introduction via: https://github.com/tensorflow/nmt
the operations are defined in this tensorflow git: 
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/seq2seq/python/ops

There is hardly any documentation.. however, the concept is (easy) to grasp. Let's start.
"""

"""
Define directories of data
"""

dir_data = r'D:\data\text\language'

"""
Load data...
"""

os.chdir(dir_data)

# Getting train data
with open('train_en.txt', encoding='utf-8') as f:
    train_text_en = f.read().splitlines()

with open('train_vi.txt', encoding='utf-8') as f:
    train_text_vi = f.read().splitlines()

# Getting test data
with open('tst2013_en.txt', encoding='utf-8') as f:
    test_text_en = f.read().splitlines()

with open('tst2013_vi.txt', encoding='utf-8') as f:
    test_text_vi = f.read().splitlines()

# Getting the vocab (top 40k words)
# These are used to set the size of the hidden layers in both LSTMs
with open('vocab_en.txt', encoding='utf-8') as f:
    vocab_en = f.read().splitlines()

with open('vocab_vi.txt', encoding='utf-8') as f:
    vocab_vi = f.read().splitlines()


# TODO Get the embedding from gensim models... for te vocab of en

"""
Prepare data
"""


class TextProcessing(object):
    """
    Here we can load text and prepare it for something like seq2seq modelling.
    Where we are able to replace words with an <unk> sign, and pre/post append <s> marks
    Also we can replace words by their integer format for making it like a lookup table.
    """
    def __init__(self, raw_text, chosen_words):
        self.raw_text = raw_text
        self.chosen_words = chosen_words
        self.dict_words = dict(zip(chosen_words, range(len(chosen_words))))
        self.start_token = '<s>'
        self.end_token = '</s>'
        self.unk_token = '<unk>'
        self.word2vec = []
        self.num_units = 0  # the amount of hidden units used in the LSTM cells (same for enc. and dec.)
        self.target_weights = []
        self.batch_size = 0
        self.max_gradient_norm = 0

    def text2int(self):
        # TODO keep in mind that this has to be optimized also for the output... Hence, keep track of the wishes and
        # requirements for the format of this
        aa = [[self.start_token] + nltk.tokenize.word_tokenize(b) + [self.end_token] for b in test_text_en[0:10]]
        bb = [[self.dict_words.get(x, self.dict_words[self.unk_token]) for x in y] for y in aa]
        bb_length = [len(x) for x in bb]
        return bb, bb_length

    def int2vec(self, sent):
        pass
        # Something like this...
        # Use an embedding.. and return that shit.
        # Maybe not that useful as a tensor thing... maybe define the graph that it is in
        # to overcome the problem of using too many variables
        # return tf.nn.embedding_lookup(self.word2vec, sent)


proc_en = TextProcessing(train_text_en, vocab_en)
int_text_en, len_text_en = proc_en.text2int()

proc_vi = TextProcessing(train_text_vi, vocab_vi)
int_text_vi, len_text_vi = proc_en.text2int()


class SequenceToSequence(object):
    """
    The input should be something like numbers...
    A vocabulary should also be given...
    """
    def __init__(self, input_train_text, output_train_text, input_vocab, output_vocab, len_input, len_output):
        self.num_units = 200
        self.embedding_size = 300
        self.src_vocab_size = len(input_vocab)
        self.tgt_vocab_size = len(output_vocab)
        self.encoder_inputs = input_train_text
        self.decoder_inputs = output_train_text
        self.src_sequence_length = len_input
        self.tgt_sequence_length = len_output

    def seq2seq_graph(self):
        # If I want to convert two graphs into one object I can use
        # tf.import_graph_def
        # Aparantly you cannot train this combined graph then
        # But some other comment said it is possible...
        # By using tf.train.import_meta_graph :)
        # Check out https://stackoverflow.com/questions/33748552/tensorflow-how-to-replace-a-node-in-a-calculation-graph
        # I think this stuff can be quite useful https://www.tensorflow.org/api_docs/python/tf/train/import_meta_graph
        # Can be used to split up graphs.. save stuff for later use.. etc..

        # Not really happy with this graph..
        graph_seq2seq = tf.Graph()

        with graph_seq2seq.as_default():
            decoder_outputs = tf.placeholder(dtype=tf.float32)
            projection_layer = tf.placeholder(dtype=tf.float32)  # TODO need to look into this.. where do we init it?

            with tf.variable_scope('seq2seq'):
                initialize = tf.initialize_all_variables()  # I think it is advices not to use this one anymore..

                # Embedding - change this tf.variable thing
                embedding_encoder = tf.Variable([self.src_vocab_size, self.embedding_size], name="embedding_encoder")
                embedding_decoder = tf.Variable([self.tgt_vocab_size, self.embedding_size], name="embedding_decoder")

                # Look up embeddings:
                encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, self.encoder_inputs)
                decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, self.decoder_inputs)

                # Build encoder and decoder RNN cell
                # TODO This only gives a single layer lstm
                # TODO maybe consider using the ntm code...? (or snippets)
                encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
                decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)

                # Helper - uses to sample from the decoder or something... avoids storing/updating a bigass matrix?
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, self.tgt_sequence_length, time_major=True)

                # Create encoder outputs
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp,
                                                                   sequence_length=self.src_sequence_length)

                # attention_states: [batch_size, max_time, num_units]
                attention_states = tf.transpose(encoder_outputs, [1, 0, 2])

                # Create an attention mechanism... from the output of the encoder of course ***
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.num_units, attention_states,
                                                                        memory_sequence_length=self.src_sequence_length)

                # Build decoder RNN cell - different set of weights than encoder ofc. But it has same hidden levels

                # Attention mechanism.. *** but also in combination with the decoder of course.
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                                   attention_layer_size=self.num_units)

                # Decoder  - we can relpace this with some Beam Search thing
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state,
                                                          output_layer=projection_layer)
                # Dynamic decoding
                # Returns: (final_outputs, final_state, final_sequence_lengths).
                outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder)

                # logits = outputs.rnn_output
                # crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits)
                # train_loss = (tf.reduce_sum(crossent * self.target_weights) / self.batch_size)

        return outputs

A = SequenceToSequence(int_text_en,int_text_vi, vocab_en, vocab_vi, len_text_en, len_text_vi)
B = A.seq2seq_graph()

# -- testing seq2seq

input_train_text = int_text_en
output_train_text = int_text_vi
input_vocab = vocab_en
output_vocab = vocab_vi
len_input = len_text_en
len_output = len_text_vi

num_units = 200
embedding_size = 300
src_vocab_size = len(input_vocab)
tgt_vocab_size = len(output_vocab)
encoder_inputs = input_train_text
decoder_inputs = output_train_text
src_sequence_length = len_input
tgt_sequence_length = len_output

# Not really happy with this graph..
###
# Here is a piece of code that can be used in order to assign various options to some output stuff
# Like using different loss functions that can easily be called ny this


def foo3(choice: str):
    return {
        'a': lambda: tf.Variable(3),
        'b': lambda: tf.Variable(6),
        'c': lambda: tf.Variable(9)
    }[choice]()

###

max_dec_out = 4
max_enc_in = 4

batch_size = 3

# TODO how to fix this... embedding look up



A = np.array(np.random.rand(10,5))
B = [[1, 2, 3, 4, 5, 6, 7], [1, 2, 4], [1, 2, 4, 3, 0, 9, 9]]
test_seq_length = [len(x) for x in B]
B1 = np.array([[1, 2, 3, 4, 5, 6, 7], [1, 2, 4, 5, 4, 2, 1], [1, 2, 4, 3, 0, 0, 2]])


A.shape


derp_graph = tf.Graph()
with derp_graph.as_default():
    A = tf.Variable(np.random.rand(1,3,3))
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(5)
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, A, dtype=tf.float64)
    B = tf.reduce_sum(encoder_outputs, name='test')

lolol = derp_graph.get_operation_by_name('test').values()[0]

derp = tf.Session(graph=derp_graph)
derp.run(tf.global_variables_initializer())
derp.run(lolol)


encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, B, sequence_length=test_seq_length, dtype=tf.float64)
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, np.array([[1,2,3],[2,2,5],[0,5,1]]), dtype=tf.float64)

derp.run(encoder_outputs)
# TODO Just use None as a size specificator


graph_seq2seq = tf.Graph()

with graph_seq2seq.as_default():
    # decoder_outputs = tf.placeholder(shape=(None, None), dtype=tf.float32)
    # projection_layer = tf.placeholder(dtype=tf.float32)  # TODO need to look into this.. where do we init it?
    # encoder_inputs = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='encoder_input')
    # decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32)

    # src_sequence_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='input_length')
    # tgt_sequence_length = tf.placeholder(shape=(None,), dtype=tf.int32)

    with tf.variable_scope('seq2seq'):
        # initialize = tf.global_variables_initializer()

        # Embedding - change this tf.variable thing
        # initialize this man
        embedding_encoder = tf.Variable(np.random.rand(src_vocab_size, embedding_size), dtype=tf.float32,
                                        name='embedding_encoder')
        # embedding_decoder = tf.Variable(np.random.rand(tgt_vocab_size, embedding_size), name="embedding_decoder",
        # dtype=tf.float32)
        # encoder_inputs = tf.Variable(np.array([[1, 2, 3, 4], [2, 3, 5]]))
        encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, encoder_inputs, name='embedding_lookup')
        # decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, decoder_inputs)

        # Build encoder and decoder RNN cell
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        # decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

        # helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, tgt_sequence_length, time_major=True)

        # Create encoder outputs
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp,
                                                           sequence_length=src_sequence_length, dtype=tf.float32)



input_dinges = np.array(input_train_text[0:N_batch])

with tf.Session(graph=graph_seq2seq) as sess:
    tf.global_variables_initializer().run()
    test = sess.run(enc_input, feed_dict={enc_input:input_dinges})

    sess.run(input_length)

enc_input = graph_seq2seq.get_operation_by_name('encoder_input').values()
input_length = graph_seq2seq.get_operation_by_name('input_length').values()
embed_enc = graph_seq2seq.get_operation_by_name('seq2seq/embedding_encoder').values()
embed_lookup = graph_seq2seq.get_operation_by_name('seq2seq/embedding_lookup').values()
embed_trans = graph_seq2seq.get_operation_by_name('seq2seq/derp').values()

N_batch = 5


###
        # attention_states: [batch_size, max_time, num_units]
        attention_states = tf.transpose(encoder_outputs, [1, 0, 2], name='derp')

        # Create an attention mechanism... from the output of the encoder of course ***
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units, attention_states,
                                                                memory_sequence_length=src_sequence_length)

        # Build decoder RNN cell - different set of weights than encoder ofc. But it has same hidden levels

        # Attention mechanism.. *** but also in combination with the decoder of course.
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                           attention_layer_size=num_units, name='decoder')

        # -- debug
        # Look up embeddings:
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state)


        # TODO This only gives a single layer lstm
        # TODO maybe consider using the ntm code...? (or snippets)


        # Helper - uses to sample from the decoder or something... avoids storing/updating a bigass matrix?

        # Decoder  - we can relpace this with some Beam Search thing
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state,
                                                  output_layer=projection_layer)
        # Dynamic decoding
        # Returns: (final_outputs, final_state, final_sequence_lengths).
        outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder)


#--- end

    def train(self, learning_rate):
        # Not sure how this fits in the framework...
        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(train_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)

        # Optimization
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

    # Append and prepend sentence blocks to it
# Make sure that unkown words are replaced

# Here I need to convert the sentences

"""
Setup some models... check if we have all the functionalities
"""



"""
Main stuff
"""





