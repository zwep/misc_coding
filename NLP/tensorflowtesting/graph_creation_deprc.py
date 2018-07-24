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
    length_op = len(graph_operations )
    print("Length of graph object: ", str(length_op ))
    if length_op:
        for i in graph_operations[:5]: print(i.outputs)
    print("id of graph object:", id(grph_obj))

def x0002_func(grph_obj):
    """Simple help function to print node names. :param grph_obj: graph object"""
    graph_operations_names = [i.name for i in grph_obj.get_operations() if not i.name.__contains__('/')]
    print(graph_operations_names)


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


tf.reset_default_graph()
print("We start out fresh.. Hence our default graph has zero operations:")
x0001_func(tf.get_default_graph())

print("Now create a standard graph")
x_input = tf.placeholder(dtype=tf.float32, name='input')
y_output = tf.placeholder(dtype=tf.float32, name='output')
w1 = tf.placeholder(dtype=tf.float32, name="weights1")
b1 = tf.Variable(2.0, name="bias1")

w2 = tf.placeholder(dtype=tf.float32, name="weights2")

# Define a test operation that we will restore
Xw1 = tf.multiply(w1, x_input, name='mult1')
Xw1b = tf.add(Xw1, b1, name='add1')

y_pred = tf.sigmoid(tf.multiply(w2, Xw1b), name='y_pred')
loss_value = tf.reduce_mean(tf.sqrt((y_output - y_pred) ** 2), name='loss_value')
optimizer = tf.train.AdamOptimizer(0.8).minimize(loss_value, name='adam_opt')

print("Checking the length of our default graph again..")
x0001_func(tf.get_default_graph())

print("We can do the same trick by defining a self created graph first")
graph_inline = tf.Graph()
print("In order to add nodes to this, we need the with-as-clause")
with graph_inline.as_default():
    x_input = tf.placeholder(dtype=tf.float32, name='input')
    y_output = tf.placeholder(dtype=tf.float32, name='output')
    z_check = tf.placeholder(dtype=tf.float32, name='check')

x0001_func(graph_inline)


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

print("But screw functions! We want Classes! But do they work any different?")


class GraphCreator:
    """
    Similar to the function
    """
    def __init__(self, graph_obj):
        self.graph_obj = graph_obj

        # Add these nodes to the graph_obj
        with graph_obj.as_default():
            x_input_fun = tf.placeholder(dtype=tf.float32, name='input')
            y_output_fun = tf.placeholder(dtype=tf.float32, name='output')


class GraphCreatorANew:
    """
    Similar to the function
    """
    def __init__(self):
        self.graph_obj = tf.Graph()

        with self.graph_obj.as_default():
            x_input_fun = tf.placeholder(dtype=tf.float32, name='input')
            y_output_fun = tf.placeholder(dtype=tf.float32, name='output')




test = tf.Graph()
one_graph_ahah = GraphCreator(test)
x0001_func(one_graph_ahah.graph_obj)
one_graph_ahah = GraphCreator(test)  # This again adds new nodes to the existing graph
x0001_func(one_graph_ahah.graph_obj)

two_graph_ahah = GraphCreatorANew()
x0001_func(two_graph_ahah.graph_obj)
two_graph_ahah = GraphCreatorANew()  # And here we have, of course, a new clean graph
x0001_func(two_graph_ahah.graph_obj)


"""
Accessing graphs and their structures

We have created some graphs in a rather silly way, without any structure or easy way to get operations back.
In this section we will describe several methods on how to retrieve operations
 
This is achieved by using
- tf.get_variables... 
- name scopes, to organize a graph...
- collections, to access and retrieve certain collections of the graph(s)
"""

# --- use get_variables to return some variables

tf.reset_default_graph()
x0001_func(tf.get_default_graph())

# With this we can either create a new variable (with name name), or get one
with tf.variable_scope('outer_scope'):
    X = tf.get_variable(name='X_var_test', dtype= tf.float32, shape=(5,2))

x0001_func(tf.get_default_graph())
with tf.variable_scope('outer_scope_2'):
    # get_variable can be usefull but generates a lot of errors when called again... can be usefull to protect your
    # graph from overloading with nodes..
    Y = tf.get_variable(name='Y_var_test', dtype= tf.float32, shape=(3,))
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
    Z = tf.get_variable(name='X_var_test', dtype= tf.float32, shape=(5,2)) + 2
    W = out_scope.get_variable(var_store=out_scope, name='X_var_test')

# Here we see what kind of operations one such get_variable creates
for i in tf.get_default_graph().get_operations():
    print(i.outputs)

tf.get_default_graph().get_operation_by_name('outer_scope/test')
tf.get_variable(name='test', dtype=tf.float32, shape=(2,5))


# --- short information about Collections

# How to add something to a collection
tf.add_to_collection(tf.GraphKeys.VARIABLES, W)
tf.get_default_graph().get_all_collection_keys()
tf.get_collection_ref(tf.GraphKeys.ACTIVATIONS)
dir(tf.GraphKeys)

tf.reset_default_graph()


w = tf.Variable([1, 2, 3], collections=[tf.GraphKeys.WEIGHTS], dtype=tf.float32, name='weight')
w2 = tf.Variable([11, 22, 32], collections=[tf.GraphKeys.WEIGHTS], dtype=tf.float32, name='weight1')
b1 = tf.Variable([1, 0, 1], dtype=tf.float32, name='bias')  # Defining a bias outside some collections

# Initialize only the Weights..
# Could check that other (no-WEIGHTS) variables are not initialized
weight_init_op = tf.variables_initializer(tf.get_collection_ref(tf.GraphKeys.WEIGHTS), name='init1')

x0001_func(tf.get_default_graph())

# Start an interactive session to avoid sess.run()
# Pretty useful to just explain things without the whole with tf.Session() as sess: sess.run() things
sess = tf.InteractiveSession()
weight_init_op.run()

try:
    b1.eval()
except tf.errors.FailedPreconditionError:
    print("We have not initialized this one")

for vari in tf.get_collection_ref(tf.GraphKeys.WEIGHTS):
    print(vari.eval())
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, 0.2 * vari)  # By this we can also add nodes to the graph

weight_update_ops = tf.get_collection_ref(tf.GraphKeys.UPDATE_OPS)

for upd in weight_update_ops:
    print(upd.eval())

for upd in weight_update_ops:
    print(upd.eval())

x0001_func(tf.get_default_graph())

sess.close()


# ---- About namescopes

# A graph with a lot of namespaces
with tf.Graph().as_default() as g:
    c = tf.constant(5.0, name="c")
    # assert c.op.name == "c"
    c_1 = tf.constant(6.0, name="c")
    # assert c_1.op.name == "c_1"

    # Creates a scope called "nested"
    with g.name_scope("nested") as scope:
        nested_c = tf.constant(10.0, name="c")
        assert nested_c.op.name == "nested/c"

        # Creates a nested scope called "inner".
        with g.name_scope("inner"):
            nested_inner_c = tf.constant(20.0, name="c")
            assert nested_inner_c.op.name == "nested/inner/c"

        # Create a nested scope called "inner_1".
        with g.name_scope("inner"):
            nested_inner_1_c = tf.constant(30.0, name="c")
            assert nested_inner_1_c.op.name == "nested/inner_1/c"

            # Treats `scope` as an absolute name scope, and
            # switches to the "nested/" scope.
            with g.name_scope(scope):
                nested_d = tf.constant(40.0, name="d")
                assert nested_d.op.name == "nested/d"

                with g.name_scope(""):
                    e = tf.constant(50.0, name="e")
                    assert e.op.name == "e"


# What exactly is the use of this..?
with tf.name_scope('jemoeder'):
    # Decalring vars here will prepend their name with 'jemoeder'

# Without the with-as-clause
cm = tf.name_scope('jemoeder')
cm.__enter__()
w = tf.Variable([1, 2, 3], collections=[tf.GraphKeys.WEIGHTS], dtype=tf.float32, name='weight')
cm.__exit__(None, None, None)
print(w.name)


# --- Now use tf.get_variable to really execute something
sess = tf.InteractiveSession()
x0001_func(tf.get_default_graph())
tes = tf.get_variable('weight1')
tf.get_variable_scope()


# -- Some idiotic Tensorboard stuff that doesnt work
# What is graph.name_scope('name')
with tf.Session(graph=tf.get_default_graph()) as sess:
    writer = tf.summary.FileWriter("D:/", sess.graph)
    writer.add_graph(tf.get_default_graph())
    writer.close()


# --- saver

tf.reset_default_graph()
# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
inc_v2 = tf.add(v1, 1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    # Do some work with the model.
    print(v1.eval())
    print(inc_v1.op.run())
    print(v1.eval())

    print(inc_v2.eval())
    dec_v2.op.run()
    # Save the variables to disk.
    save_path = saver.save(sess, r'D:\checkpoints\testcase\test.ckpt')
    print("Model saved in file: %s" % save_path)

# Restore

tf.reset_default_graph()
# Create some variables.
v1 = tf.get_variable("v1", [3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", [5], initializer = tf.zeros_initializer)

# Add ops to save and restore only `v2` using the name "v2"
saver = tf.train.Saver({"v2": v2})

# Use the saver object normally after that.
with tf.Session() as sess:
    # Initialize v1 since the saver will not.
    v1.initializer.run()
    saver.restore(sess, r'D:\checkpoints\testcase\test.ckpt')

    print("v1 : %s" % v1.eval())
    print("v2 : %s" % v2.eval())

# restore import meta grpah

saver = tf.train.Saver({"v2": v2})
with tf.Session() as sess:
#    test = tf.train.import_meta_graph(r'D:\checkpoints\testcase\test.ckpt.meta')
    # test.restore(sess, r'D:\checkpoints\testcase\test.ckpt')
    saver.restore(sess, r'D:\checkpoints\testcase\test.ckpt')
    # v1.initializer.run()
    # print("v1 : %s" % v1.eval())
    print("v2 : %s" % v2.eval())