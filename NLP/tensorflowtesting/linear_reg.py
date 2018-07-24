# Linear Regression examples
# non-Linear Regression examples

# --------------------------------------------------------------------------- #
# Load libraries
# --------------------------------------------------------------------------- #


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import imageio
import os
import glob

# --------------------------------------------------------------------------- #
# Constant
# --------------------------------------------------------------------------- #

N_accuracy = 101
N_hidden_1 = 101
N_hidden_2 = 101

reg_param = 0.0001

loc_plot = r"C:\Users\C35612.LAUNCHER\1_Data_Innovation_Analytics\code\NLP\plot" 

# --------------------------------------------------------------------------- #
# Used functions
# --------------------------------------------------------------------------- #


# Used to plot the prediction vs real values
def generate_pred_plot(x_batch, y_batch, y_pred_batch, n_epoch, i_fig=1):
    plt.figure(i_fig)
    axes = plt.gca()
    axes.set_ylim([-1, 1])
    plt.title(str(n_epoch))
    plt.scatter(x_batch[:, 0], y_batch)
    plt.plot(x_batch[:, 0], y_pred_batch[0, :], 'ro')

    # name_file = "plot/linreg" + str(N_epoch) + ".png"
    # plt.savefig(name_file)
    # plt.gcf().clear()


# Used to show the relation between iterations and loss value
def generate_error_plot(loss_val, epoch_list):
    plt.figure(1)
    plt.plot(epoch_list, loss_val)
    name_file = "plot/error_rate" + ".png"
    plt.savefig(name_file)
    plt.gcf().clear()


# Used to create a gif from the generated prediction-plots	
def generate_gif():
    images = []
    filenames = glob.glob("linreg*png")
    for filename in filenames:
        images.append(imageio.imread(filename))

    imageio.mimsave('convergence.gif', images)


# Used as input for the run
def generate_dataset():
    x_batch = np.linspace(-1, 1, N_accuracy)
    x_batch = np.reshape(x_batch, (N_accuracy, 1))

    y_batch = 2 * x_batch[:, 0] ** 3 + np.random.randn(N_accuracy) * .03
    y_batch = y_batch/np.linalg.norm(y_batch)

    return x_batch, y_batch


# Defines the neural net to execute (non)linear regression
def linear_regression():

	with tf.Graph().as_default as g_obj:
		x = tf.placeholder(tf.float64, shape=(N_accuracy, 1), name='x')
		y = tf.placeholder(tf.float64, shape=(N_accuracy,), name='y')

		with tf.variable_scope('lreg'):
			w1 = tf.Variable(np.random.normal(size=(N_accuracy, N_hidden_1)), name='W1')
			w2 = tf.Variable(np.random.normal(size=(N_hidden_1, N_hidden_2)), name='W2')
			w3 = tf.Variable(np.random.normal(size=(N_hidden_2, N_accuracy)), name='W3')

			w = list([w1, w2, w3])
			n_layer = len(w)

			b1 = tf.Variable(np.random.normal(size=(N_hidden_1,)))
			b2 = tf.Variable(np.random.normal(size=(N_hidden_2,)))

			layer_1 = tf.matmul(tf.transpose(x), w1)
			layer_1 = tf.add(layer_1, b1)
			layer_1 = tf.tanh(layer_1)

			layer_2 = tf.matmul(layer_1, w2)
			layer_2 = tf.add(layer_2, b2)
			layer_2 = tf.tanh(layer_2)

			y_pred = tf.matmul(layer_2, w3)
			# y_pred = tf.tanh(y_pred)

			loss = 0.5*tf.reduce_mean(tf.square(y_pred[0, :] - y))
			# loss = tf.reduce_mean((y_pred[0,:] - y))
			# loss = tf.reduce_sum((y_pred - y))

			# Add regularization term
			reg_term = tf.Variable(0.0)
			reg_term = tf.cast(reg_term, tf.float64)
			for i_index in range(n_layer):
				reg_term = reg_term + tf.reduce_mean(tf.nn.l2_loss(tf.cast(w[i_index], tf.float64)))

			reg_term = reg_term * reg_param
			loss = loss + reg_term

        return x, y, y_pred, loss, b1, b2, reg_term, w1, w2, w3

# Trains the neural net over several training iterations	
# Outputs the loss 


def run(n_epoch):
    x_batch, y_batch = generate_dataset()

    x, y, y_pred, loss, b1, b2, reg_term, w1, w2, w3 = linear_regression()

    # optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    optimizer = tf.train.AdamOptimizer(0.8).minimize(loss)
    weight_set_1 = []
    weight_set_2 = []
    weight_set_3 = []

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        feed_dict = {x: x_batch, y: y_batch}
        loss_value = np.array([100])
        # while loss_val > 0.11:
        for _ in range(n_epoch):
            w1_value, w2_value, w3_value, loss_value, _, y_pred_value = session.run([w1, w2, w3, loss, optimizer, y_pred], feed_dict)
            weight_set_1.append(w1_value)
            weight_set_2.append(w2_value)
            weight_set_3.append(w3_value)

        return loss_value, y_pred_value, weight_set_1, weight_set_2, weight_set_3
 
 
if __name__ == '__main__':	
    epoch_list = np.arange(100, 1000, 10)
    loss_list = np.empty(len(epoch_list))
    y_pred_list = []
    i = 1
	i_epoch = epoch_list[-1]
	
    for i in range(len(epoch_list)):
        i_epoch = epoch_list[i]
        print(i_epoch)
        loss_val, y_pred_val, weight_set_1, weight_set_2, weight_set_3 = run(10000)
        loss_list[i] = loss_val
        y_pred_list.append(y_pred_val)

	x_batch, y_batch = generate_dataset()
	generate_pred_plot(x_batch,y_batch,y_pred_val, 0)
	plt.show()
	
    # With this we just generate the error plot over the iterations
    generate_error_plot(loss_list, epoch_list)
    # With this we can generate a GIF from the created images
    os.chdir(loc_plot)
    generate_gif()

    for i in range(len(y_pred_list)):
        # plt.plot(y_pred_list[i][0])
        print(np.linalg.norm(y_pred_list[i][0]))
        # plt.plot(y_batch)
