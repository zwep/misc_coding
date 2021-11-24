import json
import os
import pickle
import re

import gensim.models
import nltk
import numpy as np
import pandas as pd

from embedding.word2vec import *


# TODO make character n-grams


def prep_oms_lines(raw_data, n_max_words=50000, window_size=3):
    """
    This is a very expensive function to prep all the data.
    Lots of for loops.
    Super ugly

    :param raw_data: content of the OMS regle in a pd.DataFrame/pd.read_csv
    :param n_max_words: max amount of words
    :param window_size: 3
    :return: a long string that can be processed by the class WordEmbedding.
    """
    clean_data = [re.sub('\s+', ' ', re.sub('^ |CCV\*', '', re.sub('[0-9]+', '', x[32:-7]))).upper() for x in raw_data]
    word_data = [nltk.tokenize.word_tokenize(x, language='dutch') for x in clean_data]
    unique_words, counts = np.unique(np.concatenate(word_data), return_counts=True)
    unique_words_sel = unique_words[counts.argsort()[-n_max_words:]]

    word_dict = dict(zip(unique_words_sel, range(len(unique_words_sel))))
    word_dict['UNKNOWN'] = len(word_dict) + 1

    word_data_int = convert_to_int(word_data, word_dict)
    # ['<' + x + '>' for b in chosen_set for x in b]  # Make sure that we separate each word..
    batch_labels = [sentence_to_label_set(x, window_size) for x in word_data_int]
    return batch_labels, word_dict, word_data


def convert_to_int(input_list, vocab_dict):
    """
    Converts input_list to their dictionary bounded value from vocab_dict

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


def sentence_to_label_set(input_sentence, window_size):
    """
    This is my own cbow/skipgram training_set generator. The implementation by tensorflow was a bit weird in my
    opinion. Besides that, it could not capture starting and ending words, which can be needed for short sentences.

    :param input_sentence: tokenized sentence
    :param window_size: how many neighbours are we considering
    :return: a list of tuples of context words and center words.
    """
    batch_label_list = []
    for i_centre in range(len(input_sentence)):
        x_centre = input_sentence[i_centre]
        window_start = i_centre - window_size
        window_end = i_centre + window_size
        window_start = window_start * int(window_start > 0)  # Makes sure that we get the right context
        window_end = window_end * int(window_end < len(input_sentence)) + len(input_sentence) * int(window_end >= len(
            input_sentence))
        x_context = input_sentence[window_start:window_end]
        x_context.remove(x_centre)
        batch_label = list(zip([x_centre] * len(x_context), x_context))
        batch_label_list.append(batch_label)
    return batch_label_list


"""
locations
"""

location_text = r"C:\Users\C35612.LAUNCHER\Testing_data\Word2vec"
location_wiki = r"C:\Users\C35612.LAUNCHER\Testing_data\Word2vec\wikipedia"
location_plot = r"C:\Users\C35612.LAUNCHER\1_Data_Innovation_Analytics\code\NLP\plot"
location_oms = r'D:\data\text\OMSregel'

os.chdir(location_oms)
A = pd.read_csv('raw_OMS_regel_10000.csv')
# A = pd.read_csv('raw_OMS_regel_10000000.csv')
A = A['oms_regel_1']

A_dataset, A_dictionary, A_sentences = prep_oms_lines(A)
dum = [x for b in A_dataset for x in b]  # ugly solution in order to fix the fact that the stuff is nested
A_trainingset = [x for b in dum for x in b]  # tada... training set

json.dump(A_dictionary, open('dict_OMS_10000000.txt', 'w'))
pickle.dump(A_trainingset, open('labels_OMS_10000000.p', 'wb'))


"""
Checking stuff with gensim models.
gensim.models.doc2vec.FAST_VERSION > -1  # use this piece of code to check if there is a C interpreter
Becaues the C interepreter makes the implementation 70x faster.
"""

model = gensim.models.Word2Vec(A_sentences, size=200)
# Dit werkt zo verdomd goed heh
# sg=1  gives skipgram

model = gensim.models.Word2Vec(A_sentences, min_count=10, iter=20, sg=1, size=300, alpha=0.2)
# Kijk dit zijn best goede dingen lijkt mij
model.similarity('ALBERT', 'HEIJN')


"""
If we have executed the code above... we save it as a json/dump thing and then we can use it again somewhere else to 
train stuff..
"""

os.chdir(location_oms)
vocab_dict_oms = json.load(open("dict_OMS_10000000.txt", 'r'))
label_oms = pickle.load(open('labels_OMS_10000000.p', 'rb'))

batch_size = 500
n_subset = 200000
index_subset = np.random.randint(0, len(label_oms), n_subset)
label_oms_subset = label_oms[index_subset]
label_oms_subset = label_oms[:n_subset]
# Try to find out a way to execute Epocs
# test = SkipGram(label_oms, vocab_dict_oms, batch_size)
test = SkipGram(label_oms_subset, vocab_dict_oms, batch_size)

test.vocabulary_size += 1  # I really dont know why, has to do with some lookup thing
n_epoch = 100
test.n_steps *= n_epoch

a, b = test.run(0.6)
a_plot = [x[0] for x in a]
plt.plot(a_plot)
a = test.get_batch()
a1, b1 = list(zip(*a))


tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=1000, method='exact')
tsne_document = tsne.fit_transform(b.embedding[0:nrange, :])

plt.figure(figsize=(9, 9))  # in inches
for i_index in range(nrange):
    x, y = tsne_document[i_index, :]
    plt.scatter(x, y)
    plt.annotate(xy=(x, y), s=b.vocab_dict[i_index])

plt.show()
