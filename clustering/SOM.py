# encoding: utf-8

"""
In this file we only define the SOM functionality... adding some comments to make it more readable.

origin: https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
But I added more comments and examples myself
"""

import tensorflow as tf
import numpy as np


class SOM(object):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function and linearly decreasing learning rate.
    Note that each neuron/node in the 2-D Map has a high dimension. It is (AFAIK) a discrete clustering algorithm,
    where each new vector sits in a place on the 2-D Map where it is closest to that node.
    However, because you can train this Map with various examples.. it has some power.
    """

    # Class variable to check if the SOM has been trained
    _trained = False

    def __init__(self, m, n, dim, n_iterations=100, alpha=None, sigma=None):
        """
        Initializes all necessary components

        :param m:  dimension of the SOM
        :param n:  dimension of the SOM
        :param dim: the dimensionality of the training inputs.
        :param n_iterations: should be an integer denoting the number of iterations undergone
        while training.
        :param alpha: a number denoting the initial time(iteration no)-based learning rate. Default value is 0.3
        :param sigma:  the initial neighbourhood value, denoting the radius of influence of the BMU while training.
        By default, its taken to be half of max(m, n).
        """

        # Assign required variables first
        self._m = m
        self._n = n
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))

        self._weightages = []  # Pre-allocating vector weights
        self._locations = []  # Pre-allocating vector locations
        self._centroid_grid = []  # Pre-allocating vector with centroids

        # Initialize graph
        self._graph = tf.Graph()

        # Build components in the graph.
        with self._graph.as_default():
            """
            Variables and Constants operations for data storage
            """

            # Randomly initialized weightage vectors for all neurons/nodes, stored together as a
            #  matrix Variable of size [m*n, dim]
            self._weightage_vects = tf.Variable(tf.random_normal([m*n, dim]))

            # Matrix of size [m*n, 2] for SOM grid locations of neurons
            # Example output of _neuron_locations: [[0,0],[0,1],[1,0],[1,1]]
            self._location_vects = tf.constant(np.array(list(self._neuron_locations(m, n))))

            """
            Placeholder for training inputs
            """
            # The training vector
            self._vect_input = tf.placeholder(dtype='float', shape=[dim], name='vector_input')
            # Iteration number
            self._iter_input = tf.placeholder(dtype='float', name='iter_input')

            """
            Construct training operation piece by piece
            """
            # Only the final, 'root' training operation needs to be assigned as an attribute to self, since all the
            # rest will be executed automatically during training. This is the idea of the graph in tensorflow.. you
            # need one node that is linked to the rest

            # To compute the Best Matching Unit (BMU) given a vector basically calculates the Euclidean distance
            # between every neuron's weightage vector and the input, and returns the index of the neuron which gives
            # the least value.
            _matrix_input = tf.stack([self._vect_input for i in range(m*n)])  # Duplicate input vec to SOM-size
            _matrix_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self._weightage_vects, _matrix_input), 2), 1))
            bmu_index = tf.argmin(_matrix_dist, 0)

            # This will extract the location of the BMU based on the BMU's index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]), np.array([[0, 1]]))  # Since the index ranges from 1 to m*n
            slice_input = tf.cast(slice_input, tf.int32)  # Make sure that we have ints as locations..
            # Here we slice and reshape _locations_vects, keep in mind that the following things are equivalent..
            # t = tf.constant([[[1, 1, 1], [2, 2, 2]],
            # [[3, 3, 3], [4, 4, 4]],
            # [[5, 5, 5], [6, 6, 6]]])
            # tf.slice(t, [1, 0, 0], [1, 2, 3])
            # A = tf.Session().run(t)
            # A[1:2,0:2,0:3]  this is the more known way of slicing
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input, tf.constant(np.array([1, 2]))), [2])

            # To compute the alpha and sigma values based on iteration number
            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input, self._n_iterations))  # LR = 1 - (i/N)
            _alpha_op = tf.multiply(alpha, learning_rate_op)  # alpha * LR
            _sigma_op = tf.multiply(sigma, learning_rate_op)  # sigma * LR

            # Construct the op that will generate a vector with learning rates for all neurons, based on iteration
            # number and location wrt BMU.
            # This takes into account that only an area around the BMU will have a certain update, based on sigma
            # You can see it as a Gaussian distribution around the BMU that is being updated.
            _matrix_bmu_loc = tf.stack([bmu_loc for i in range(m*n)])
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(self._location_vects, _matrix_bmu_loc), 2), 1)
            bmu_distance_squares = tf.cast(bmu_distance_squares, "float32")
            neighbourhood_func = tf.exp(tf.negative(tf.div(bmu_distance_squares, tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)

            # Finally, the op that will use learning_rate_op to update the weightage vectors of all neurons based on
            # a particular input
            # The tf.tile function allows you to replicate the tensor in certain dimensions. Here we are replicating
            # the values across the dimension dim
            tiled_lrng = [tf.tile(tf.slice(learning_rate_op, np.array([i]), np.array([1])), [dim]) for i in range(m*n)]
            learning_rate_multiplier = tf.stack(tiled_lrng)

            diff_input_weight = tf.subtract(tf.stack([self._vect_input for i in range(m*n)]), self._weightage_vects)
            weightage_delta = tf.multiply(learning_rate_multiplier, diff_input_weight)
            new_weightages_op = tf.add(self._weightage_vects, weightage_delta)  # Here we add the delta to the weights
            self._training_op = tf.assign(self._weightage_vects, new_weightages_op)  # Create new op

            """
            Initialize session
            """
            self._sess = tf.Session()

            """
            Initialize variables
            """
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

    @staticmethod
    def _neuron_locations(m, n):
        """
        Yields one by one the 2-D locations of the individual neurons in the SOM. By using the keyword yield we
        create a generator out of this.
        """
        # Nested iterations over both dimensions to generate all 2-D locations in the map
        # We could use list comprehension here I guess.. But this is cleaner
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with dimensionality (dim) as provided during
        initialization of this SOM. Current weightage vectors for all neurons(initially random) are taken as starting
        conditions for training.
        """

        # Training iterations
        # Train with each vector one by one
        for iter_no in range(self._n_iterations):
            for input_vect in input_vects:
                feed_dict = {self._vect_input: input_vect, self._iter_input: iter_no}
                self._sess.run(self._training_op, feed_dict=feed_dict)

        # Store a centroid grid for easy retrieval later on
        # It seems that this is essentially just a reshape of the weights-vector.
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid

        self._trained = True

    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid

    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """

        if not self._trained:
            raise ValueError("SOM not trained yet")

        to_return = []
        for vect in input_vects:
            list_index = [i for i in range(len(self._weightages))]
            min_index = min(list_index, key=lambda x: np.linalg.norm(vect - self._weightages[x]))
            to_return.append(self._locations[min_index])

        return to_return
