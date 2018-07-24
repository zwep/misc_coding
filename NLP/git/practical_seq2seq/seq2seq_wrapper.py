# -*- encoding: utf-8 -*-

"""
Mario: It's a strange world

Got this from
https://github.com/suriyadeepan/practical_seq2seq

"""

import tensorflow as tf
import numpy as np
import sys

# This is needed in order to fix some error wtf which error then....
# Because somehow there is a problem with deepcopy...
#
setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)


class Seq2Seq(object):
    """
    A simple Seq2Seq object.
    """
    def __init__(self, xseq_len, yseq_len, xvocab_size, yvocab_size, emb_dim, num_layers, ckpt_path, lr=0.0001,
                 epochs=100, model_name='model', save_step=50):

        # attach these arguments to self
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.ckpt_path = ckpt_path
        self.epochs = epochs
        self.model_name = model_name
        self.save_step = save_step

        # build thy graph
        #  attach any part of the graph that needs to be exposed, to the self
        def __graph__():
            # placeholders
            tf.reset_default_graph()  # Why not use with tf.Graph as ... ?
            #  encoder inputs : list of indices of length xseq_len
            self.enc_ip = [tf.placeholder(shape=[None, ],
                                          dtype=tf.int64,
                                          name='ei_{}'.format(t)) for t in range(xseq_len)]

            #  labels that represent the real outputs
            self.labels = [tf.placeholder(shape=[None, ],
                                          dtype=tf.int64,
                                          name='derp_{}'.format(t)) for t in range(yseq_len)]

            #  decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
            self.dec_ip = [tf.zeros_like(self.enc_ip[0], dtype=tf.int64, name='GO')] + self.labels[:-1]

            # Basic LSTM cell wrapped in Dropout Wrapper
            self.keep_prob = tf.placeholder(tf.float32)
            # define the basic cell
            basic_cell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(emb_dim, state_is_tuple=True),
                output_keep_prob=self.keep_prob)
            # stack cells together : n layered model
            stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([basic_cell] * num_layers, state_is_tuple=True)

            # for parameter sharing between training model
            #  and testing model
            with tf.variable_scope('decoder') as scope:
                # build the seq2seq model
                #  inputs : encoder, decoder inputs, LSTM cell type, vocabulary sizes, embedding dimensions
                self.decode_outputs, self.decode_states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(self.enc_ip,
                                                                                                          self.dec_ip,
                                                                                                          stacked_lstm,
                                                                                                          xvocab_size,
                                                                                                          yvocab_size,
                                                                                                          emb_dim)
                # share parameters
                scope.reuse_variables()
                # testing model, where output of previous timestep is fed as input
                #  to the next timestep
                self.decode_outputs_test, self.decode_states_test = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                    self.enc_ip, self.dec_ip, stacked_lstm, xvocab_size, yvocab_size, emb_dim,
                    feed_previous=True)

            # now, for training,
            #  build loss function

            # weighted loss
            #  TODO : add parameter hint
            loss_weights = [tf.ones_like(label, dtype=tf.float32) for label in self.labels]
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.decode_outputs, self.labels, loss_weights,
                                                                yvocab_size)
            # train op to minimize the loss
            self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        sys.stdout.write('<log> Building Graph ')
        # build comput graph
        __graph__()
        sys.stdout.write('</log>')

    '''
        Training and Evaluation
    '''

    # get the feed dictionary
    def get_feed(self, mat_x, mat_y, keep_prob):
        """
        Returns a dictionary with the mapping between tensors and values
        :param mat_x: x-seq values
        :param mat_y: y-seq values
        :param keep_prob: some dropout value
        :return:
        """
        feed_dict = {self.enc_ip[t]: mat_x[t] for t in range(self.xseq_len)}
        feed_dict.update({self.labels[t]: mat_y[t] for t in range(self.yseq_len)})
        feed_dict[self.keep_prob] = keep_prob  # dropout prob
        return feed_dict

    # run one batch for training
    def train_batch(self, sess, train_batch_gen):
        """
        Run a train batch with a given session and train_batch generator

        :param sess: A valid session
        :param train_batch_gen: A generator creating a batch for x and y values
        :return: the loss value of the run
        """
        # get batches
        batch_x, batch_y = train_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed(batch_x, batch_y, keep_prob=0.5)
        _, loss_v = sess.run([self.train_op, self.loss], feed_dict)
        return loss_v

    def eval_step(self, sess, eval_batch_gen):
        """
        A way to validate the current model parameters
        :param sess: A valid session
        :param eval_batch_gen: A generator that creates validation checks
        :return:
        """
        # get batches
        batch_x, batch_y = eval_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed(batch_x, batch_y, keep_prob=1.)
        loss_v, dec_op_v = sess.run([self.loss, self.decode_outputs_test], feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1, 0, 2])
        return loss_v, dec_op_v, batch_x, batch_y

    # evaluate 'num_batches' batches
    def eval_batches(self, sess, eval_batch_gen, num_batches):
        """
        Here we can evaluate multiple batches with the eval-batch-generators

        :param sess:
        :param eval_batch_gen:
        :param num_batches:
        :return:
        """
        losses = []
        for i in range(num_batches):
            loss_v, dec_op_v, batch_x, batch_y = self.eval_step(sess, eval_batch_gen)
            losses.append(loss_v)
        return np.mean(losses)

    # finally the train function that
    # runs the train_op in a session
    # evaluates on valid set periodically
    # prints statistics
    def train(self, train_set, valid_set, sess=None):
        """
        Training the model

        :param train_set:
        :param valid_set:
        :param sess:
        :return:
        """
        # we need to save the model periodically
        saver = tf.train.Saver()

        # if no session is given
        if sess is None:
            print("No session\n")
            # create a session
            sess = tf.Session()
            # init all variables
            sess.run(tf.global_variables_initializer())
        else:
            print("I have a session\n")

        sys.stdout.write('\n<log> Training started </log>\n')
        # run M epochs
        for i in range(self.epochs):
            print("Current i: "+str(i))
            try:
                self.train_batch(sess, train_set)
                if i and i % self.save_step == 0:  # TODO : make this tunable by the user
                    # save model to disk
                    saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
                    # saver.save(sess, 'my_test_model', global_step=1000)
                    # evaluate to get validation loss
                    val_loss = self.eval_batches(sess, valid_set, 16)  # TODO : and this
                    # print stats
                    print('\nModel saved to disk at iteration #{}'.format(i))
                    print('val   loss : {0:.6f}'.format(val_loss))
                    sys.stdout.flush()
            except KeyboardInterrupt:  # this will most definitely happen, so handle it
                print('Interrupted by user at iteration {}'.format(i))
                self.session = sess
                return sess

    def restore_last_session(self, i_step):
        """
        Here we can restore the last session. A change has happened here due to the new
        tensorflow version... We now load the old checkpoint by using tf.train.import_meta_graph

        https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model

        :param i_step: This is the step we want to get
        :return: recovered session based on the checkpoint
        """
        # saver = tf.train.Saver()
        assert i_step % self.save_step == 0
        meta_name = self.ckpt_path + '/' + self.model_name + '.ckpt-' + str(i_step) + '.meta'
        saver = tf.train.import_meta_graph(meta_name)

        # create a session in which we can load the checkpoint values
        sess = tf.Session()
        # get checkpoint state

        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)

        # restore session if both variables are not empty
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            # return to user
            return sess

    # prediction
    def predict(self, sess, mat_x):
        """

        :param sess:
        :param mat_x:
        :return: an encoded version of the prediction, based on our dict
        """
        feed_dict = {self.enc_ip[t]: mat_x[t] for t in range(self.xseq_len)}
        feed_dict[self.keep_prob] = 1.
        dec_op_v = sess.run(self.decode_outputs_test, feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1, 0, 2])
        # return the index of item with highest probability
        return np.argmax(dec_op_v, axis=2)
