# encoding: utf-8

"""
Here we try to run some dummy model using the code available from

git.tensorflow_seq2seq_tutorials

But now I see that I need an older version of tensorflow for this.
So Im going to translate that... tomorrow.
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell
from git.tensorflow_seq2seq_tutorials.model_new import Seq2SeqModel, train_on_copy_task
import git.tensorflow_seq2seq_tutorials.helpers as helpers

tf.reset_default_graph()
tf.set_random_seed(1)

with tf.Session() as session:

    # with bidirectional encoder, decoder state size should be
    # 2x encoder state size
    model = Seq2SeqModel(encoder_cell=LSTMCell(10),
                         decoder_cell=LSTMCell(20),
                         vocab_size=10,
                         embedding_size=10,
                         attention=True,
                         bidirectional=True,
                         debug=False)

    session.run(tf.global_variables_initializer())

    train_on_copy_task(session, model,
                       length_from=3, length_to=8,
                       vocab_lower=2, vocab_upper=10,
                       batch_size=100,
                       max_batches=3000,
                       batches_in_epoch=1000,
                       verbose=True)