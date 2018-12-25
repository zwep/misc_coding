# encoding: utf-8

"""

Here we have a bunch of video-functions that we use in other programmes...

Tralala

"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sklearn.linear_model


def n_diff_cluster(input_list, n, return_index=False):
    """
    Groups a list based on the difference between the value n.

    By default returns the clustered values instead of the index

    :param input_list: list of integers
    :param return_index: option to get either index or values
    :param n: the allowed difference between consecutive elements
    :return:
    """

    prev = None
    temp_group = []
    for i, x in enumerate(input_list):
        if not prev or (x - prev) == n:
            if return_index:
                temp_group.extend([i])
            else:
                temp_group.extend([input_list[i]])
        else:
            yield list(temp_group)
            if return_index:
                temp_group = [i]
            else:
                temp_group = [input_list[i]]
        prev = x
    if temp_group:
        yield list(temp_group)


def find_ngrams(input_list, n):
    """
     Because its awesome
    :param input_list: list of items
    :param n: size of ngram
    :return: ngram
    """
    return zip(*[input_list[i:] for i in range(n)])


def transform_to_color(x):
    """
    Based on the amount of unique points in the label list...
    Return a list that contains a translation to distinct colors
    """
    unique_x = list(set(x))
    n = len(unique_x)
    color_x = list(iter(plt.cm.rainbow(linspace(0, 1, n))))
    dict_color = dict(zip(unique_x, color_x))
    x_to_color = list(map(dict_color.get, x))
    return x_to_color


def diff_list(input_list):
    """
    Returns the difference of a list

    :param input_list: list with numeric values
    :return: list with numeric values, but
    """
    return [j-i for i, j in zip(input_list[:-1], input_list[1:])]


def color_back_red(x, x_text):
    """
    Formats the background of a string

    :param x: input text
    :param x_text: piece of text that needs to be highlighted
    :return: string that, when printed, shows the color red.
    """
    import colorama
    x_split = x.replace(x_text, ' ::: ').split(':::')
    x_red_text = colorama.Back.RED + x_text + colorama.Style.NORMAL
    return x_red_text.join(x_split)


def color_back_red_index(x, index):
    """Formats the background of a string by using index instead of text"""
    import colorama
    list_x = list(x)
    for i_index in index:
        list_x[i_index] = colorama.Back.RED + list_x[i_index] + colorama.Style.NORMAL

    return ''.join(list_x)


def subset_list(y, n_min):
    """
    Subsets a list by looking at the length of each element.

    :param y: input list of list
    :param n_min: the minimum length for a list in the lists
    :return: index of the succesful  list, and value of them
    """
    y_index = []
    y_sub = []
    y_index_sub = [(i, x) for i, x in enumerate(y) if len(x) > n_min]
    if len(y_index_sub) != 0:
        # This here is needed to unzip the tuple and map it to two lists.
        y_index, y_sub = list(map(list, zip(*y_index_sub)))
    return y_index, y_sub


def linspace(a, b, n):
    """
    Exactly same functionality as np.linspace.

    :param a: starting point
    :param b: ending point
    :param n: length of the list..
    :return: list with length n
    """
    return [a + x*(b-a)/(n-1) for x in range(n)]


def dict_find(key, dictionary):
    """
    Used to find the values of a certain key in a nested dict

    :param key:
    :param dictionary:
    :return:
    """
    for k, v in dictionary.items():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in dict_find(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                for result in dict_find(key, d):
                    yield result


def read_txt_file(path, n_files=None):
    """
    This loads .txt files specified in the :param path: value

    :param path: location of the data
    :param n_files: the amount of reviews to be loaded
    :return: list of strings

    TODO review if the removing of html tags in permitted in here
    """
    import re
    import glob

    intent_path = path + '\*.txt'
    list_files = glob.glob(intent_path)
    sub_list_files = list_files[0:n_files]
    # Removes html tags... maybe not put this here
    content_files = [re.sub('<.*>', ' ', y) for x in sub_list_files for y in open(x, encoding='utf-8').readlines()]
    return content_files


def get_item_occurence(input_list):
    """
    Counts the occurence of subsequence elements while preserving the order in which they happened

    e.g.
    x = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'a', 'a', 'c']
    y = get_item_occurence(x)
    print(y)
        [('a', 4), ('b', 3), ('a', 2), ('c', 1)]
    :param input_list: input list with string items
    :return: list with tuples
    """
    i_count = 1
    n_class = 0
    count_classes = []
    group_classes = [n_class]
    temp_classes = input_list[0]

    for i_class in input_list[1:] + ['stop']:
        if i_class == temp_classes:
            group_classes.append(n_class)
            i_count += 1
        else:
            n_class += 1
            group_classes.append(n_class)
            count_classes.append((temp_classes, i_count))
            i_count = 1

            temp_classes = i_class

    return count_classes, group_classes[:-1]  # Second last because of the 'stop' trigger
