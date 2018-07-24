# encoding: utf-8

"""
Presenting all the plot functionalities that we are using

"""

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_dbscan(db):
    """
    Plotting a db scan thing.

    :param db:
    :return:
    """
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in for k, col in zip(unique_labels, colors): if k == -1:
    # Black used for noise.
        col = [0, 0, 0, 1]
    #
    class_member_mask = (labels == k)
    #
    xy = data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)
    #
    xy = data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)
    #
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def plot_som(som, data):
    #Get output grid
    # This fucking sucks
    # use the funciont get_mapped_vect or something
    import numpy as np
    import pandas as pd

    image_grid = som.get_centroids()
    z = np.asarray(image_grid)
    #Map colours to their closest neurons
    mapped = som.map_vects(data)
    #
    position_list = list()
    for i in range(len(embed)):
        single_word = embed[i,:]
        result = np.dot(z,single_word)
        dinges = np.unravel_index(result.argmax(),result.shape)
        position_list.append(dinges)
    #
    x1 = np.array(position_list)
    x2 = np.array(list(words_dict_reverse.values()))
    x2 = np.reshape(x2,(len(x2),1))
    derpderderp = np.append(x1,x2,axis = 1)
    A = pd.DataFrame(derpderderp)
    A.columns = ['x','y','name']
    #
    A = A.set_index(['x','y'])
    print(A.sort_index())


def tsne_plot(data, color):
    """
    Caluclate TSNE clustering
    :param data:
    :param color:
    :return:
    """
    tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=1000, method='exact')
    tsne_document = tsne.fit_transform(data)

    plt.figure(figsize=(9, 9))  # in inches
    for i_index, i_color in enumerate(color):
        x, y = tsne_document[i_index, :]
        plt.scatter(x, y, color=i_color)

    plt.show()


def generate_error_plot(loss_value, epoch_list):
    """
    Used to show the relation between iterations and loss value
    :param loss_value:
    :param epoch_list:
    :return:
    """
    plt.figure(1)
    plt.plot(epoch_list, loss_value)
    name_file = "plot_error_rate" + ".png"
    plt.savefig(name_file)
    plt.gcf().clear()


def generate_gif():
    """
    Used to create a gif from the generated prediction-plots
    :return:
    """
    import glob
    import imageio

    images = []
    filenames = glob.glob("linreg*png")
    for filename in filenames:
        images.append(imageio.imread(filename))

    imageio.mimsave('convergence.gif', images)


def plot_with_labels(low_dim_embs, labels, filename):
    """
    Function to draw visualization of distance between embeddings.

    :param low_dim_embs:
    :param labels:
    :param filename:
    :return:
    """
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)
