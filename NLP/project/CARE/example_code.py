#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Code for seq2seq model x

Using mainly the model defined in

git.practical_seq2seq

Pretty neat
"""

import os
# os.chdir('C:/Users/C35612.LAUNCHER/IdeaProjects/NLP/')  # Used so I can copy it easily to Python Console

import re
import tensorflow as tf

# preprocessed data
from git.practical_seq2seq.datasets.twitter import data as twitter_data
import git.practical_seq2seq.data_utils as data_utils

import git.practical_seq2seq.seq2seq_wrapper as seq2seq_wrapper

from importlib import reload  # This one is super useful when you want to reload a class


"""
Model saving things
"""


def x0001_func(grph_obj):
    """ Simple help function. :param grph_obj: graph object"""
    graph_operations = grph_obj.get_operations()
    length_op = len(graph_operations)
    print("\n Length of graph object: ", str(length_op))
    if length_op:
        for i in graph_operations[:5]:
            print(i.outputs)
    print("id of graph object:", id(grph_obj), "\n")


def x0002_func(grph_obj):
    """Simple help function to print node names. :param grph_obj: graph object"""
    graph_operations_names = [i.name for i in grph_obj.get_operations() if not i.name.__contains__('/')]
    print('\n graph operation names', graph_operations_names)


"""
Set config values
"""

# load data from pickle and npy files
metadata, idx_q, idx_a = twitter_data.load_data(PATH=r'D:/data/text/Twitter/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 1
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 10
epochs = 11
save_step = 100
num_layers = 1
ckpt_path = 'D:/data/text/Twitter/checkpoints/'
model_name = 'second_try'

"""
Set data generators
"""

val_batch_gen = data_utils.rand_batch_gen(validX, validY, 32)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)
test_batch_gen = data_utils.rand_batch_gen(testX, testY, 256)

"""
Init Model
"""

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                                yseq_len=yseq_len,
                                xvocab_size=xvocab_size,
                                yvocab_size=yvocab_size,
                                ckpt_path=ckpt_path,
                                model_name=model_name,
                                emb_dim=emb_dim,
                                num_layers=num_layers,
                                epochs=epochs,
                                save_step=save_step)


"""
Train the model
"""

sess = model.train(train_batch_gen, val_batch_gen)


"""
Restart training from checkpoint / Test from checkpoint
"""
# http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/


tf.reset_default_graph()  # Make sure that the default graph is empty
# Restart training session/load checkpoint
i_step = 2000
model_ckpt_path = ckpt_path + model_name + '.ckpt-' + str(i_step)
meta_name = model_ckpt_path + '.meta'


tf.reset_default_graph()
# Create a clean graph and import the MetaGraphDef nodes.
graph_restore_2 = tf.Graph()

with graph_restore_2.as_default():
    saver = tf.train.import_meta_graph(meta_name)

with tf.Session(graph=graph_restore_2) as sess_restore_2:
    saver.restore(sess_restore_2, model_ckpt_path)

    x_in, y_in = train_batch_gen.__next__()
    feed_dict = model.get_feed(x_in, y_in, keep_prob=0.5)
    sess_restore_2.run(tf.get_default_graph().get_operation_by_name('Adam'), feed_dict=feed_dict)


tf.reset_default_graph()
# Create a clean graph and import the MetaGraphDef nodes.
graph_restore_2 = tf.Graph()
with tf.Session(graph=graph_restore_2) as sess_restore_2:
    model.train(train_batch_gen, val_batch_gen, sess=sess_restore_2)
    # Import the previously export meta graph.
    saver = tf.train.import_meta_graph(meta_name)
    saver.restore(sess_restore_2, ckpt.model_checkpoint_path)
    model.train(train_batch_gen, val_batch_gen, sess=sess_restore_2)


# create a session in which we can load the checkpoint values
sess_restore = tf.Session()
saver.restore(sess_restore, ckpt.model_checkpoint_path)

# Recreating part of the graph...
graph_restore = tf.get_default_graph()
model.train(train_batch_gen, val_batch_gen, sess=sess_restore)
[x for x in graph_restore.get_operations() if re.findall('laceholder', x.type)]
[x for x in sess_restore.graph.get_operations() if re.findall('laceholder', x.type)]
[x for x in tf.get_default_graph().get_operations() if re.findall('laceholder', x.type)]
[x for x in tf.get_default_graph().get_operations() if re.findall('ariable', x.type)]


# Find what is in the graph
graph_restore.get_operations()
graph_placeholders = [x for x in tf.get_default_graph().get_operations() if re.findall('laceholder', x.type)]
print(graph_placeholders)

# Get all the tensors/operations from the graph
# But I am sure that we should also be able to obtain this from the model-object
enc_ip_rec = [graph_restore.get_tensor_by_name('ei_{}:0'.format(t)) for t in range(xseq_len)]
dec_ip_rec = [graph_restore.get_tensor_by_name('derp_{}:0'.format(t)) for t in range(yseq_len)]
keep_prob = graph_restore.get_tensor_by_name('Placeholder:0')
adam_op_rec = graph_restore.get_operation_by_name('Adam')

# Training part of the graph...
# Here we start the training again
# Creation of the feeddict
batch_x, batch_y = train_batch_gen.__next__()
feed_dict = {enc_ip_rec[t]: batch_x[t] for t in range(xseq_len)}
feed_dict.update({dec_ip_rec[t]: batch_y[t] for t in range(yseq_len)})
feed_dict.update({keep_prob: 0.5})
# Run the training
sess_restore.run(adam_op_rec, feed_dict)

# Now try to run this restore session with the model boject


"""
Test the model from last session method
"""

i_step = 2000
sess_test = model.restore_last_session(i_step)
input_ = test_batch_gen.__next__()[0]
output = model.predict(sess_test, input_)
print(output.shape)

# Print out the replies of the output compared to the input
# Also applied id-to-word conversion
replies = []
for ii, oi in zip(input_.T, output):
    q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
    decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
    if decoded.count('unk') == 0:
        if decoded not in replies:
            print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
            replies.append(decoded)
