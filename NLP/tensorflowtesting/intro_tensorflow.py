# --------------------------------------------------------------------------- #
# Load libraries
# --------------------------------------------------------------------------- #

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Define
# --------------------------------------------------------------------------- #

N_h = 100
N_x = 784
N_y = 100

def create_graph():
	lr = 0.5

	g_obj = tf.Graph()
	with g_obj.as_default():
		x = tf.placeholder(tf.float32, (N_h,N_x))
		y = tf.placeholder(tf.float32, (N_y,1))
		
		b = tf.Variable(tf.zeros((N_h,1)))
		W = tf.Variable(tf.random_uniform((N_h,N_h),-1,1))
		
		h = tf.nn.softmax(tf.matmul(W,x) + b)
		prediction = tf.reduce_mean(h, reduction_indices = [1])
		loss = tf.reduce_mean(tf.squared_difference(y, prediction))
		train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
		
	return g_obj, train_step, x, y

the_graph, train, x_in, y_out = create_graph()

'''
#Fetches
List of graph nodes
Return the outputs of these nodes

#Feeds
Dictionary mapping from graph nodes to concrete values.
Specifies the value of each grap node given in the dictionary
'''

# Start sessions on default setup.
# Could add GPU thingsh ere
sess = tf.Session()
# Lazy evaluation
#sess.run(tf.initialize_all_variables())
sess.run(tf.global_variables_initializer())
for i in range(100):
	res = sess.run(train_step,{x: np.random.random((N_b,N_W))})

#Variables: parameters for tuning
#placeholder: input

plt.imshow(res)


# ---
# Easy example
# ---

# --------------------------------------------------------------------------- #
# Define
# --------------------------------------------------------------------------- #
# Shows how easy this is...
# Because this is just applying ReLU to some matrix vector mult.
N_b = 1
N_W = 2

b = tf.Variable(tf.zeros(N_b))
#W = tf.Variable(tf.random_uniform((N_W,N_b),-1,1))
W = tf.Variable(np.matrix([[2.0,1],[1,-2]]))
W = tf.to_float(W)
x = tf.placeholder(tf.float32, (N_b,N_W))
h = tf.nn.relu(tf.matmul(x,W) + b)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
x_hat = np.matrix([1,2])
res = sess.run(h,{x: x_hat})
plt.imshow(res)
