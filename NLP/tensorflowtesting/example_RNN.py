from colour import Color
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


"""
 ---------------------- Functions ---------------------- 
"""


def generate_dataset(total_time):
    x = np.linspace(0, 2*np.pi, total_time+1)
    y = np.sin(x)
    return x, y


def generate_batch_random(n_size):
    x = np.reshape(np.random.rand(n_size), (n_size, 1))*1000
    y = np.reshape(np.ones(n_size), (n_size, 1))
    return x, y


def generate_batch(x, global_index, n_input):
    new_index = global_index + n_input
    x_batch = x[global_index: global_index + n_input],
    return new_index, x_batch


def simple_network(learning_rate):
    n_input = 10
    g1 = tf.Graph()

    with g1.as_default() as g:
        ph_batch_x = tf.placeholder(shape=(n_input, 1), dtype=tf.float64, name='ph_x')
        ph_batch_y = tf.placeholder(shape=(n_input, 1), dtype=tf.float64, name='ph_y')

        with tf.variable_scope('rnn'):
            w_softmax = tf.Variable(np.random.rand(n_input, n_input), dtype=tf.float64, name='weights')
            b_softmax = tf.Variable(np.zeros((n_input, 1)), dtype=tf.float64, name='bias')

            y_pred = tf.add(tf.matmul(w_softmax, ph_batch_x), b_softmax, name='pred')

            losses = tf.subtract(y_pred, ph_batch_y, name='loss')
            squared_losses = tf.square(losses)
            total_loss = 0.5*tf.reduce_mean(squared_losses, name='total_loss')

            abs_losses = tf.abs(losses)  # We can also use absolute loss...
            # total_loss = 0.5*tf.reduce_mean(abs_losses, name='total_loss')

            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, name='optimizer')
            # optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss, name='optimizer')
    return g1


# sigma(W x_t + U h + b)
# :param W: this is the weight matrix with size n_hidden x n_ftr
# :param x_t: this is the input matrix with size n_ftr x 1
# :param U: this is the weight matrix with size n_hidden x n_hidden
# :param h: this is the state matrix with size n_hidden x 1
# :param b: this is the bias matrix with size n_hidden x 1
def rnn_network(learning_rate, batch_size, n_seq, n_ftr, n_hidden):
    g1 = tf.Graph()

    with g1.as_default():
        ph_batch_x = tf.placeholder(shape=(batch_size, n_seq * n_ftr), dtype=tf.float64, name='ph_x')
        ph_batch_y = tf.placeholder(shape=(batch_size, n_seq), dtype=tf.float64, name='ph_y')

        with tf.variable_scope('rnn'):
            w_softmax = tf.Variable(np.random.rand(n_hidden, 1), dtype=tf.float64, name='w_softmax')
            b_softmax = tf.Variable(np.zeros((batch_size, 1)))

            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
            # rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden, reuse=True if is_first_step else None)
            # Unpack columns
            input_series = tf.split(ph_batch_x, n_seq, axis=1, name='input_series')
            label_series = tf.unstack(ph_batch_y, axis=1, name='label_series')

            states_series, current_state = tf.nn.static_rnn(rnn_cell, inputs=input_series, dtype=tf.float64)
            logits_series = tf.add(tf.matmul(current_state, w_softmax), b_softmax, name='logits')
            # predictions_series = tf.nn.softmax(logits_series, name='pred')  # HAHAHA THIS IS THIS BIGGEST JOKE
            predictions_series = logits_series
            actual_value = tf.reshape(label_series[-1], (batch_size, 1), name='actual')
            losses = tf.subtract(predictions_series, actual_value, name='loss')
            squared_losses = tf.square(losses)
            total_loss = 0.5*tf.reduce_mean(squared_losses, name='total_loss')
            optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss, name='optimizer')
            # optimizer = tf.train.AdamOptimizer(0.3).minimize(total_loss, name='optimizer')
    return g1


# W_softmax = tf.Variable(np.random.rand(n_hidden, num_classes), dtype=tf.float64)
# b_softmax = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float64)
# logits_series = [tf.matmul(state, W_softmax) + b_softmax for state in states_series]  # Broadcasted addition
# predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
# losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(
#     logits_series, labels_series)]
# total_loss = tf.reduce_mean(losses)
def run_simple_network():
    simple_graph = simple_network(0.1)
    n_input = 10
    n_epoch = 10000
    n_plot = 100
    n_color = n_epoch // n_plot

    red = Color("red")
    colors = list(red.range_to(Color("green"), n_color))
    counter = 0

    input_x, output_y = generate_batch_random(n_input)

    ph_x = simple_graph.get_tensor_by_name('ph_x:0')
    ph_y = simple_graph.get_tensor_by_name('ph_y:0')
    pred = simple_graph.get_tensor_by_name('rnn/pred:0')
    loss_stuff = simple_graph.get_tensor_by_name('rnn/total_loss:0')
    optimizer = simple_graph.get_operation_by_name('rnn/optimizer')

    feed_dict = {ph_x: input_x, ph_y: output_y}

    with tf.Session(graph=simple_graph) as sess:
        tf.global_variables_initializer().run()
        for i_step in range(n_epoch):
            sess.run(optimizer, feed_dict=feed_dict)
            if i_step % n_plot == 0:
                loss_value = sess.run(loss_stuff, feed_dict=feed_dict)
                pred_value = sess.run(pred, feed_dict=feed_dict)
                plt.plot(pred_value, color=colors[counter].get_rgb())
                plt.scatter(range(len(output_y)), output_y)
                print(loss_value)
                counter += 1


def run_rnn_network():

    n_epoch = 100
    n_plot = 10
    n_color = n_epoch // n_plot
# Define colors that can be used to plot stuff
    red = Color("red")
    colors = list(red.range_to(Color("green"), n_color))

    n_size = 100
    batch_size = 5
    n_seq = 20
    num_batch = int(np.ceil(n_size/batch_size/n_seq))  # This is the amount of times we send a batch...

    learning_rate = 0.8
    n_hidden = 20
    n_ftr = 1
    input_x, output_y = generate_dataset(n_size)

    rnn_graph = rnn_network(learning_rate, batch_size, n_seq, n_ftr, n_hidden)

    # Get the content from the graph that we want to run/plot/track
    ph_batch_x = rnn_graph.get_operation_by_name('ph_x').values()
    ph_batch_y = rnn_graph.get_operation_by_name('ph_y').values()
    logits_value = rnn_graph.get_operation_by_name('rnn/logits').values()
    actual_value = rnn_graph.get_operation_by_name('rnn/actual').values()
    total_loss = rnn_graph.get_operation_by_name('rnn/total_loss').values()
    optimizer = rnn_graph.get_operation_by_name('rnn/optimizer')

    loss_value_list = []

    for i_ in range(n_epoch):
        g_index_x = 0  # This is the global index that is used to generate the batches...
        g_index_y = 0  # This is the global index that is used to generate the batches...

        for j_ in range(num_batch):
            g_index_x, batch_x = generate_batch(input_x, g_index_x, n_seq * batch_size)
            g_index_y, batch_y = generate_batch(output_y, g_index_y, n_seq * batch_size)

            if len(batch_x[0]) == n_seq:
                derp_x = np.reshape(batch_x[0], (batch_size, n_seq))
                derp_y = np.reshape(batch_y[0], (batch_size, n_seq))
            else:
                diff_length_x = n_seq - len(batch_x[0])
                diff_length_y = n_seq - len(batch_y[0])
                derp_x = np.reshape(np.append(batch_x[0], np.zeros(diff_length_x)), (batch_size, n_seq))
                derp_y = np.reshape(np.append(batch_y[0], np.zeros(diff_length_y)), (batch_size, n_seq))

            feed_dict = {ph_batch_x: derp_x, ph_batch_y: derp_y}

            with tf.Session(graph=rnn_graph) as sess:
                tf.global_variables_initializer().run()
                sess.run(optimizer, feed_dict=feed_dict)
                loss_value = sess.run(total_loss, feed_dict=feed_dict)
                loss_value_list.append(loss_value)
