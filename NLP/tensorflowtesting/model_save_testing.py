# encoding: utf-8

"""
Here we are trying to recreate the model saving, loading and re-training part
"""

import numpy as np
import tensorflow as tf


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
Creating graphs

In this section we will show you the different ways on creating graph structures:
- Inline, by using a with-as-clause
- Inline, by NOT using a with-as-clause
- In functions, by either sending or creating a tf.Graph() in the function
- In classes, similar to functions of course

tl;dr; Creating graphs in separated parts (like functions or classes) can allow you 
to create a modular graph structures. Where certain functions are responsible for
specific parts of a graph.
"""

# <editor-fold desc='Some initial graph using default_graph'>

tf.reset_default_graph()
print("We start out fresh.. Hence our default graph has zero operations:")
x0001_func(tf.get_default_graph())

print("Now create a standard graph")
x_input = tf.placeholder(dtype=tf.float32, name='input')
y_output = tf.placeholder(dtype=tf.float32, name='output')
w1 = tf.placeholder(dtype=tf.float32, name="weights1")
b1 = tf.Variable(2.0, name="bias1")

print("Checking the length of our default graph again..")
x0001_func(tf.get_default_graph())

# </editor-fold>

# <editor-fold desc='Some initial graph using tf.Graph()'>

print("We can do the same trick by defining a self created graph first")
graph_inline = tf.Graph()
print("In order to add nodes to this, we need the with-as-clause")
with graph_inline.as_default():
    x_input = tf.placeholder(dtype=tf.float32, name='input')
    y_output = tf.placeholder(dtype=tf.float32, name='output')
    z_check = tf.placeholder(dtype=tf.float32, name='check')

x0001_func(graph_inline)

# </editor-fold>

# <editor-fold desc='Some initial graph using __enter__ command'>

print("We can also do this without the with-as-clause, like so:")
graph_cmd = tf.Graph()
cm = graph_cmd.as_default()
cm.__enter__()

print("We can even check this by looking at their id now")
x0001_func(graph_cmd)
x0001_func(tf.get_default_graph())

# All nodes are added to the `graph` now
x = tf.placeholder(dtype=tf.float32, name='x')
y = tf.placeholder(dtype=tf.float32, name='y')

# Exit the context
cm.__exit__(None, None, None)

print("And again, we check that we have exited it the right way")
x0001_func(graph_cmd)
x0001_func(tf.get_default_graph())

# </editor-fold>

# <editor-fold desc='Some initial graph using a function'>

print("We can also create a function that returns a graph object")

def create_graph(graph_obj):
    """
    Adds two new placeholdesr to the graph_obj
    :param graph_obj:
    :return:
    """
    with graph_obj.as_default() as g_def:
        x_input_fun = tf.placeholder(dtype=tf.float32, name='input')
        y_output_fun = tf.placeholder(dtype=tf.float32, name='output')

    # return g_def

def create_graph_anew():
    """
    create a graph new every time
    :return:
    """
    graph_obj = tf.Graph()
    with graph_obj.as_default() as g_def:
        x_input_fun = tf.placeholder(dtype=tf.float32, name='input')
        y_output_fun = tf.placeholder(dtype=tf.float32, name='output')
        tf.Variable(0.5, dtype=tf.float32)

    return g_def

print("Using an empty graph, we can add two nodes using our new function")
graph_func = tf.Graph()
x0001_func(graph_func)
create_graph(graph_func)  # We dont even need to return a graph object
x0001_func(graph_func)  # Becaues all the nodes are added due to the with-as-clause

print("Does this adding of nodes also work for existing graphs?")
x0001_func(graph_inline)
create_graph(graph_inline)  # Even existing graphs can be altered
x0001_func(graph_inline)  # To add some nodes..

print("Creating a whole new graph")
graph_func_new = create_graph_anew()  # Now we do need to assign a variable to it
x0001_func(graph_func_new)
print("And also here, of course, we can add new nodes")
create_graph(graph_func_new)  # But if we want to add two extra nodes
x0001_func(graph_func_new)

# </editor-fold>

# <editor-fold desc='Some inital graph using Classes'>

print("But screw functions! We want Classes! But do they work any different?\n")


class SimpleGraphCreator:
    """
    Some simple graph object...
    """
    def __init__(self):
        self.graph_obj = tf.Graph()

        with self.graph_obj.as_default() as g_def:
            x_input_fun = tf.placeholder(dtype=tf.float32, name='input')
            y_output_fun = tf.placeholder(dtype=tf.float32, name='output')
            w_weights_fun = tf.get_variable('weight_set', dtype=tf.float32, shape=(5, 5))
            output = tf.matmul(x_input_fun, w_weights_fun, name='pred')
            loss = tf.subtract(output, y_output_fun, name='loss')
            opti = tf.train.AdamOptimizer().minimize(loss, name='opti')
            saver = tf.train.Saver()  # Putting this more 'up' in the graph, will change the content that the saver
            # has! See an example of this below...

        self.graph_obj = g_def


print("The answer is no. They work similar to functions..")

tf.reset_default_graph()
test = tf.Graph()
one_graph_ahah = SimpleGraphCreator()
x0001_func(one_graph_ahah.graph_obj)  # All our operations are in the class' graph object
x0001_func(tf.get_default_graph())  # Just to check.. nothing has been written to the default graph

one_graph_ahah = SimpleGraphCreator()  # And because we re-create the graph everytime...
x0001_func(one_graph_ahah.graph_obj)  # ... We dont add any extra operations in it

# </editor-fold>


"""
Save and restore a Graph, then accessing graphs and retrain!
 
In the above example we have defined a rather simple Graph... the SimpleGraphCreator... now we want to plug in some 
values, try to train a bit, save the model... bring it back and continue training. But also see how the weights 
change... 
So we are going to dig into using tf.Session() and tf.Graph() etc. and all the questions that come along
"""

# <editor-fold desc='Saving and restoring a pre-made graph under normal circumstances'>

# Create a standard graph...
standard_graph = SimpleGraphCreator().graph_obj

# Having a session and a pre-defined graph...
# And save the graph...
with tf.Session(graph=standard_graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, 'D:/checkpoints/test_case/normal_case')  # This makes a checkpoint file, data, and metagraph file


# Now reload the situation (8-)) and try to get some outputs
# Note: We are not specifying any graph.. so the session will use the default graph to store everything

n_epoch = 10
tf.reset_default_graph()  # Just make sure it is empty
with tf.Session() as sess_restore:
    normal_saver = tf.train.import_meta_graph('D:/checkpoints/test_case/normal_case.meta')
    normal_saver.restore(sess_restore, 'D:/checkpoints/test_case/normal_case')

    x0001_func(sess_restore.graph)  # An overview of what we have loaded
    x0002_func(sess_restore.graph)  # incase you forget the names of the nodes you made

    train_op = tf.get_default_graph().get_operation_by_name('opti')  # Get the optimizer back
    pred_value = tf.get_default_graph().get_operation_by_name('pred').outputs[0]  # If we want to check the pred.
    loss_value = tf.get_default_graph().get_operation_by_name('loss').outputs[0]  # If we want to plot the 'loss'
    model_weights = tf.get_default_graph().get_operation_by_name('weight_set').outputs[0]  # The weights

    x = tf.get_default_graph().get_operation_by_name('input').outputs[0]  # With this we can refer back to the input
    y = tf.get_default_graph().get_operation_by_name('output').outputs[0]  # With this we can refer back to the output
    x_feed = np.random.rand(5, 5)  # Some random input
    print("Input\n")
    print(x_feed)
    y_feed = np.array([1, 0, 0, 0, 0])  # Some random output
    in_feed_dict = {x: x_feed, y: y_feed}  # And a dict..
    print("\nWeights from the model \n")
    print((sess_restore.run(model_weights)))  # initial weight norm
    print(np.linalg.norm(sess_restore.run(model_weights)))  # initial weight norm
    for i in range(n_epoch):
        print("---------------------", i, "---------------------")
        # run the model
        pred_val, thing_loss, _ = sess_restore.run([pred_value, loss_value, train_op], feed_dict=in_feed_dict)
        print("\n Prediction value \n")
        print(pred_val)
        print("\nWeights from the model after training \n")
        print((sess_restore.run(model_weights)))  # after running
        print(np.linalg.norm(sess_restore.run(model_weights)))  # after running


# Even when running this block of code multiple times, we see that the loss stays the same..
# However... the random input changes, and everything else as well.. but no matter what.. the loss stays the same
# under these conditions. Quite funny, apparantly it is a maximal adaption or something

print("After this we have that the default graph is equal to the imported graph")
print("This is really '", tf.get_default_graph() == sess_restore.graph, "'")
print("This is because we did not specify any (predefined) graph for the session")

# Quick note: tf.train.Saver() can only be initiazed in an environment where the default graph contains saveable
# variables, like weights and biases. Not placeholders of course

# Note/question: does the position of the saver determine what he rememebers? -> Trying out import and export
# graph method, restore it in a session and check the graph operations -> Yes it matter

# Note/question: how can we access multiple checkpoints when only one checkpoint file is created? Refering,
# or restoring to the checkpoint is not necessary. If you give the base-name of the saved model with the (relative)
# directory, you are safe as well.

# </editor-fold>

"""
Playing around with the save.Saver object, and multiple graphs in one directory

- showing the impact on tf.get_default_graph() when loading new graphs
- using multiple savers in one graph... showing that the position matters
- showing how to use collections efficiently in a high-tech graph
"""

# <editor-fold desc='Showing that the position of tf.train.Saver() is important!'>


class SaverGraphCreator:
    """
    A graph object with multiple tf.train.Saver() nodes
    """
    def __init__(self):
        self.graph_obj = tf.Graph()
        self.saver_end = tf.train.Saver(defer_build=True)  # This creates a saver object without building it

        with self.graph_obj.as_default() as g_def:
            w_weights_fun = tf.get_variable('weight_set', dtype=tf.float32, shape=(5, 5))
            self.saver_begin = tf.train.Saver()  # The first saver, cant go higher, needs some variables
            x_input_fun = tf.placeholder(dtype=tf.float32, name='input')
            y_output_fun = tf.placeholder(dtype=tf.float32, name='output')
            output = tf.matmul(x_input_fun, w_weights_fun, name='pred')
            self.saver_middle = tf.train.Saver()  # The middle saver
            loss = tf.subtract(output, y_output_fun, name='loss')
            opti = tf.train.AdamOptimizer().minimize(loss, name='opti')
            self.saver_end.build()  # = tf.train.Saver(defer_build=True)  # The final saver - is only build when
            # needed, this option is great for flexibility and visibility.

        try:
            self.saver_outside = tf.train.Saver()
        except ValueError:
            print("The default graph is empty!:", x0001_func(tf.get_default_graph()))

        self.graph_obj = g_def


# Create the saver-graph object
saver_graph_object = SaverGraphCreator()

# Make sure that the default graph is empty
tf.reset_default_graph()
x0001_func(tf.get_default_graph())

# Now initialize all the variables of the graph.. and save the savers
with tf.Session(graph=saver_graph_object.graph_obj) as sess:
    sess.run(tf.global_variables_initializer())
    saver_graph_object.saver_begin.save(sess, 'D:/checkpoints/test_case/jemoederisdik_begin', global_step=1)
    saver_graph_object.saver_middle.save(sess, 'D:/checkpoints/test_case/jemoederisdik_middle', global_step=1)
    saver_graph_object.saver_end.save(sess, 'D:/checkpoints/test_case/jemoederisdik_end', global_step=1)


# Quik note: When you save several sessions... there will be only one checkpoint generated..
# However, you can always retrieve this by giving the specified path

# Restore the three saves and check their content
with tf.Session() as sess_restore:
    saver_begin = tf.train.import_meta_graph('D:/checkpoints/test_case/jemoederisdik_begin-1.meta')
    saver_begin.restore(sess_restore, 'D:/checkpoints/test_case/jemoederisdik_begin-1')
    x0001_func(sess_restore.graph)
    print(tf.get_default_graph() == sess_restore.graph)

x0001_func(tf.get_default_graph())  # We see the default graph increasing

with tf.Session() as sess_restore:
    saver_begin = tf.train.import_meta_graph('D:/checkpoints/test_case/jemoederisdik_middle-1.meta')
    saver_begin.restore(sess_restore, 'D:/checkpoints/test_case/jemoederisdik_middle-1')
    x0001_func(sess_restore.graph)
    print(tf.get_default_graph() == sess_restore.graph)

x0001_func(tf.get_default_graph())  # Even more nodes are being added!

with tf.Session() as sess_restore:
    saver_begin = tf.train.import_meta_graph('D:/checkpoints/test_case/jemoederisdik_end-1.meta')
    saver_begin.restore(sess_restore, 'D:/checkpoints/test_case/jemoederisdik_end-1')
    x0001_func(sess_restore.graph)
    print(tf.get_default_graph() == sess_restore.graph)

x0001_func(tf.get_default_graph())  # And this contains the most nodes.. This is because we did not specify any
# inital graph for the sessions... also using tf.get_default_graph() inside a with-Session-statement, will make sure
# that we get the default graph IN that session.


# </editor-fold>



# <editor-fold desc='Modular graph build'>
# Here we try to make graph creation more modular... and understand it therefore yourself better
# But this takes quite some time. So I might postpone it, but it is a good exercise

def add_some_layer(x_inp, var_scope, scope_reuse=False, n_hid=10, graph_obj=tf.Graph()):
    """

    :param x_inp:
    :param var_scope:
    :param scope_reuse:
    :param n_hid:
    :param graph_obj:
    :return:
    """
    with graph_obj.as_default():
        with tf.variable_scope(var_scope, reuse=scope_reuse):
            w_set = tf.get_variable('weight_set', dtype=tf.float32, shape=(n_hid, x.shape[1]))
            b = tf.get_variable('bias_set', dtype=tf.float32, shape=(n_hid,1))
            new_layer_op = tf.add(tf.matmul(x_inp, w_set), b)
    return new_layer_op

graph_obj = tf.Graph()
with graph_obj.as_default():
    x_test_input = tf.placeholder(dtype=tf.float32, shape=(10,10), name='input')

tes = add_some_layer(x_test_input,'first_layer', graph_obj=graph_obj)

# </editor-fold>

"""
Using collections inside a graph to easily retrieve operations 
"""

# <editor-fold desc='Creating a graph that uses collections to easily retrieve operations'>


class CollectionsGraphCreator:
    """
    A graph object with multiple nodes that are added to collections
    """
    def __init__(self):
        self.graph_obj = tf.Graph()
        self.saver_obj = tf.train.Saver(defer_build=True)

        with self.graph_obj.as_default() as g_def:
            with tf.variable_scope('first_set') as l_one:
                w1 = tf.get_variable('weight_set', dtype=tf.float32, shape=(5, 5))
            with tf.variable_scope('second_set') as l_two:
                w2 = tf.get_variable('weight_set', dtype=tf.float32, shape=(5, 5))

            b_bias_fun = tf.get_variable('bias_set', dtype=tf.float32, shape=(5, 5))
            x_input_fun = tf.placeholder(dtype=tf.float32, name='input')
            y_output_fun = tf.placeholder(dtype=tf.float32, name='output')
            output = tf.add(tf.matmul(x_input_fun, w1), b_bias_fun, name='pred')

            loss = tf.subtract(output, y_output_fun, name='loss')
            opti = tf.train.AdamOptimizer().minimize(loss, name='opti')  # want to add this one to it as well...

            # Add more variables to some collection - This is pretty inefficient and ugly-looking
            tf.add_to_collection('input variables', x_input_fun)
            tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, y_output_fun)  # This does not work either
            tf.add_to_collection('input variables', y_output_fun)
            # tf.add_to_collection(tf.GraphKeys.WEIGHTS, [w1, w2])  # This doesnt work
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, w1)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, w2)
            tf.add_to_collection(tf.GraphKeys.BIASES, b_bias_fun)
            self.saver_obj.build()

        self.graph_obj = g_def

# How to save a graph object with Collections addede to it?


# Create the Collections-graph object
coll_graph_object = CollectionsGraphCreator().graph_obj

coll_graph_object.get_collection(tf.GraphKeys.WEIGHTS)
coll_graph_object.get_collection(tf.GraphKeys.WEIGHTS, scope='first_set')
coll_graph_object.get_collection(tf.GraphKeys.WEIGHTS, scope='second_set')


# Now initialize all the variables of the graph.. and save the savers
with tf.Session(graph=coll_graph_object) as sess:
    sess.run(tf.global_variables_initializer())
    saver_graph_object.saver_begin.save(sess, 'D:/checkpoints/test_case/collection_test', global_step=1)


with tf.Session() as sess_restore:
    saver_begin = tf.train.import_meta_graph('D:/checkpoints/test_case/collection_test-1.meta')
    saver_begin.restore(sess_restore, 'D:/checkpoints/test_case/collection_test-1')
    x0001_func(sess_restore.graph)
    print(tf.get_default_graph() == sess_restore.graph)

# With this we can group certain operations...
# Can be convenient, see:
# https://stackoverflow.com/questions/34235557/what-is-the-purpose-of-graph-collections-in-tensorflow

# </editor-fold>


"""
Here we have an example of using the variable_scope argument.

Not that exciting.. since it only pre-pends some name to an existing variable. The only useful thing of this, 
is that we can set the variable-share option on or off.
"""

tf.reset_default_graph()
x0001_func(tf.get_default_graph())

# With this we can either create a new variable (with name name), or get one
with tf.variable_scope('outer_scope'):
    X = tf.get_variable(name='X_var_test', dtype=tf.float32, shape=(5, 2))

x0001_func(tf.get_default_graph())
with tf.variable_scope('outer_scope_2'):
    # get_variable can be usefull but generates a lot of errors when called again... can be usefull to protect your
    # graph from overloading with nodes..
    Y = tf.get_variable(name='Y_var_test', dtype=tf.float32, shape=(3,))
    W = tf.Variable([1, 2, 3], collections=[tf.GraphKeys.WEIGHTS], dtype=tf.float32, name='weight')
    V = tf.placeholder(dtype=tf.float32, name='first_placeholder')


tf.global_variables()
tf.local_variables()
x0001_func(tf.get_default_graph())
# Get name_scope of graph object

with tf.Graph().as_default() as def_gr:

    with tf.variable_scope('in_scope') as in_scope:
        print(in_scope.original_name_scope)
        print(in_scope.name)
        print(def_gr.get_name_scope())

    with tf.variable_scope('outer_scope', reuse=True) as out_scope:
        print(out_scope.original_name_scope)
        print(out_scope.name)
        print(def_gr.get_name_scope())

    print(out_scope.trainable_variables())
    Z = tf.get_variable(name='X_var_test', dtype=tf.float32, shape=(5, 2)) + 2
    W = out_scope.get_variable(var_store=out_scope, name='X_var_test')

# Here we see what kind of operations one such get_variable creates
for i in tf.get_default_graph().get_operations():
    print(i.outputs)

tf.get_default_graph().get_operation_by_name('outer_scope/test')
tf.get_variable(name='test', dtype=tf.float32, shape=(2, 5))
