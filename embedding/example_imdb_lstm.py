# http://deeplearning.net/tutorial/lstm.html

# can also try to use doc2vec
# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb

import nltk
import tensorflow as tf
import numpy as np
from colour import Color
import matplotlib.pyplot as plt

import glob


"""
Define functions
"""
location_imbd = r'D:\data\Imdb_real_one\train'


def load_data(path, n_reviews, intent='pos'):
    """
    Well this of course loads all the data...

    :param path: location of the data
    :param n_reviews: the amount you want to load
    :type intent: 'pos' or 'neg'
    :param intent: positive or negative reviews?
    :return: the data. DUH. Label is obvious because of intent
    """
    intent_path = path + '\\' + intent + '\*.txt'
    list_reviews = glob.glob(intent_path)
    glob.glob(r'D:\*')
    sub_list_reviews = list_reviews[0:n_reviews]
    content_review = [open(x).readlines() for x in sub_list_reviews]
    return content_review


def prep_oms_lines(clean_data, n_max_words=50000, window_size=3):
    """
    This is a very expensive function to prep all the data.
    Lots of for loops.
    Super ugly

    :param clean_data: list of strings
    :param n_max_words: max amount of words
    :param window_size: 3
    :return: a long string that can be processed by the class WordEmbedding.
    """
    word_data = [nltk.tokenize.word_tokenize(x, language='dutch') for x in clean_data]
    unique_words, counts = np.unique(np.concatenate(word_data), return_counts=True)
    unique_words_sel = unique_words[counts.argsort()[-n_max_words:]]

    word_dict = dict(zip(unique_words_sel, range(len(unique_words_sel))))
    word_dict['UNKNOWN'] = len(word_dict) + 1
    word_dict['<END>'] = len(word_dict) + 1

    word_data_int = convert_to_int(word_data, word_dict)

    return word_data_int, word_dict


def convert_to_int(input_list, vocab_dict):
    """
    Converts input_list to their dicionary bounded value from vocab_dict

    :param input_list: Can be either a list with words, or a list of list with words
    :param vocab_dict: dictionary should contain atleast the couple UNKNOWN
    :return: the integer version of the list based on the dictionary
    """
    chosen_set_int = []
    for i_set in input_list:
        if isinstance(i_set, str):
            chosen_set_int.append(vocab_dict.get(i_set, vocab_dict['UNKNOWN']))
        else:
            chosen_set_int.append([vocab_dict.get(x, vocab_dict['UNKNOWN']) for x in i_set])
    return chosen_set_int


def get_batch(data, data_index, batch_size):
    """
     When a full set of input/labels is given. We can retrieve it with this one.
    """
    assert len(data) > 0
    start_index = data_index
    end_index = data_index + batch_size
    data_index = end_index
    input_x, output_y = map(list, zip(*data[start_index:end_index]))
    return input_x, output_y, data_index


def lstm_network(learning_rate, batch_size, n_seq, n_ftr, n_hidden):
    """
    This function creates the (simple) lstm graph that is needed for the prediction
    :return: a graph object
    """
    graph_lstm = tf.Graph()

    with graph_lstm.as_default() as g:
        ph_batch_x = tf.placeholder(shape=(batch_size, n_seq * n_ftr), dtype=tf.float64, name='ph_x')
        ph_batch_y = tf.placeholder(shape=batch_size, dtype=tf.float64, name='ph_y')

        with tf.variable_scope('lstm'):
            w_logit = tf.Variable(np.random.rand(n_hidden, 1), dtype=tf.float64, name='w_logit')
            b_logit = tf.Variable(np.zeros((batch_size, 1)))

            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
            # rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden, reuse=True if is_first_step else None)
            # Unpack columns
            input_series = tf.split(ph_batch_x, n_seq, axis=1, name='input_series')
            label_series = ph_batch_y  # tf.unstack(ph_batch_y, axis=1, name='label_series')

            states_series, current_state = tf.nn.static_rnn(rnn_cell, inputs=input_series, dtype=tf.float64)
            agg_mean_series = tf.reduce_mean(states_series, axis=0, name='checkthis')
            linear_reg_series = tf.add(tf.matmul(agg_mean_series, w_logit), b_logit, name='logits')
            predictions_series = tf.sigmoid(linear_reg_series)
            actual_value = tf.reshape(label_series[-1], (batch_size, 1), name='actual')
            losses = tf.subtract(predictions_series, actual_value, name='loss')
            squared_losses = tf.square(losses)
            total_loss = 0.5*tf.reduce_mean(squared_losses, name='total_loss')
            optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss, name='optimizer')
    return graph_lstm


def run_lstm(lstm_graph, data, n_batch):
    """
    This runs one epoch over defined lstm network
    :param lstm_graph: the provided tensorflow graph
    :param data: all the data of course
    :param n_batch: amount of batches that we need
    :return:
    """
    counter = 0

    ph_x = lstm_graph.get_operation_by_name('ph_x').values()
    ph_y = lstm_graph.get_operation_by_name('ph_y').values()
    pred = lstm_graph.get_operation_by_name('lstm/logits').values()
    loss_stuff = lstm_graph.get_operation_by_name('lstm/total_loss').values()
    optimizer = lstm_graph.get_operation_by_name('lstm/optimizer')
    local_batch_size = int(ph_x[0].shape[0])
    local_nmax_seq = int(ph_y[0].shape[1])

    with tf.Session(graph=lstm_graph) as sess:
        tf.global_variables_initializer().run()
        for i_step in range(n_batch):
            input_x, output_y, new_counter = get_batch(data, data_index=counter, batch_size=local_batch_size)
            input_x = np.reshape(input_x[0] + [0] * (local_nmax_seq - len(input_x[0])), (local_batch_size,
                                                                                         local_nmax_seq))

            counter = new_counter
            feed_dict = {ph_x: input_x, ph_y: output_y}
            sess.run(optimizer, feed_dict=feed_dict)
            if (n_batch // 10) % i_step == 0:
                loss_value = sess.run(loss_stuff, feed_dict=feed_dict)
                pred_value = sess.run(pred, feed_dict=feed_dict)
                # plt.plot(pred_value, color=colors[counter].get_rgb())
                # plt.scatter(range(len(output_y)), output_y)
                print(loss_value, pred_value)
                counter += 1
        return lstm_graph


def main():
    n_articles = 10
    pos_review = load_data(location_imbd, n_articles, intent='pos')  # Simple example of how the data is loaded.
    neg_review = load_data(location_imbd, n_articles, intent='neg')  # Simple example of how the data is loaded.
    pos_review = [x[0] for x in pos_review]  # Unpack the list one level
    neg_review = [y[0] for y in neg_review]  # Unpack the list one level

    input_data = pos_review + neg_review
    target_data = [1] * n_articles + [0] * n_articles

    input_data_int, data_dict = prep_oms_lines(input_data)  # Convert the strings to ints and get a translation
    # dictionary

    all_data = list(zip(input_data_int, target_data))

    n_seq_max = np.max([len(x) for x in input_data_int])
    n_hidden = 50
    batch_size = 1
    n_batch = n_articles * 2 // batch_size
    n_ftr = 1
    # Now we might consider using dynamic rnn..
    lstm_graph = lstm_network(learning_rate=0.5, batch_size=batch_size, n_seq=n_seq_max, n_ftr=n_ftr, n_hidden=n_hidden)

    derp = run_lstm(lstm_graph, all_data, n_batch)

    input_x, output_y, new_counter = get_batch(all_data, data_index=0, batch_size=1)

if __name__ == '__main__':
    main()
