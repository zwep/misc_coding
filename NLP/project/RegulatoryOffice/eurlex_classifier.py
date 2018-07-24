# encoding: utf-8

"""
Here we are going to classify the documents based on either Impact on Business or Impact on Function


"""

import requests
import importlib
import nltk
from helper.plotfunction import tsne_plot
import itertools
import numpy as np
from helper.miscfunction import color_back_red, n_diff_cluster
import textprocessing.processtext as proc_text

import glob
import os
import re
import gensim
import matplotlib.pyplot as plt

from gensim.models.doc2vec import TaggedDocument
import project.RegulatoryOffice.fileexplorer as reg_fe
import project.RegulatoryOffice.ridfunctions as proc_rid

from nltk.tokenize.treebank import TreebankWordTokenizer
from sklearn.manifold import TSNE
from helper.miscfunction import transform_to_color


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


class WordTokenizeBracket(TreebankWordTokenizer):
    """
    Adaption of the TreebankWordTokenizer which is used by nltk.tokenize.word_tokenize

    Trying to treat '(a)' as one entity instead of ['(', 'a', ')']
    """
    def __init__(self):
        self.PARENS_BRACKETS = [(re.compile(r'(\([0-9a-zA-Z]{1,3}\))'), r' \g<0> ')]
        self.lhs_p = re.compile(r'[\(\[{]')
        self.rhs_p = re.compile(r'[\)\]}]')

    def tokenize_bonus(self, text, convert_parentheses=False, return_str=False):
        """
        Bonus in parenthesis handeling
        :param text:
        :param convert_parentheses:
        :param return_str:
        :return:
        """
        z = self.tokenize(text, convert_parentheses, return_str)
        # Replace those items that contain only one parenthesis..
        z = [self.rhs_p.sub(' \g<0>', self.lhs_p.sub('\g<0> ', x))
             if bool(self.lhs_p.search(x)) != bool(self.rhs_p.search(x)) else x for x in z]

        # Join and split again..
        return ' '.join(z).split()


own_tokenizer = WordTokenizeBracket()

# <editor-fold desc='Define locations'>
dir_data = r'D:\temp'
dir_reg_doc = dir_data + r'\reg_doc'
dir_reg_html = dir_reg_doc + r'\html'
name_data = 'Export_RID_16012018.xlsx'

# </editor-fold>

# <editor-fold desc='Load RID data'>
rid_data = proc_rid.get_RID(name_data, dir_reg_doc)  # Gives back a dictionary
A = proc_rid.ExtractRidData(glob.glob(dir_reg_html + '\\*.html'), rid_data)
# </editor-fold>

# <editor-fold desc='Load html objects'>
html_obj = proc_rid.load_html_obj(dir_reg_html)  # Gives back a tuple of (content, name)
text_dict = {}
for i_obj in html_obj:
    temp = reg_fe.ExtractRegDoc(*i_obj, rid_data).input_class_true
    temp_text = [re.sub('\n+', '\n', x.text) for x in temp if re.search('normal', x['class'][0], re.IGNORECASE)]
    A = proc_text.CleanText(temp_text)
    temp_text = A.remove_stopwords_text().lemmatize_text().text
    temp_text_joined = ' '.join([x for x in temp_text if len(x) > 4])
    temp_dict = {i_obj[1]: temp_text_joined}
    text_dict.update(temp_dict)
# </editor-fold>

# <editor-fold desc='Load target data'>
target_business_dict = {}
for x in html_obj:
    dest_temp = A.prep_rid_data[x[1]]['Impact on Functions (2nd LOD)']
    temp = {x[1]: [os.path.basename(x) for x in dest_temp]}
    target_business_dict.update(temp)
# </editor-fold>

# <editor-fold desc='Apply doc2vec for feature creation'>
doc2_vec_list = []
z = []
for i_obj, i_value in text_dict.items():
    # Get the target label(s)...
    target_list = target_business_dict[i_obj]
    for i_target in target_list:
        # print(i_target)
        z.append(i_target)
        # Here we can decide what is going in...
        tokenized_words = own_tokenizer.tokenize_bonus(i_value)
        # tokenized_words = nltk.tokenize.word_tokenize(i_value)
        temp = TaggedDocument(tokenized_words, [i_target])
        # print(temp)
        doc2_vec_list.append(temp)


model = gensim.models.doc2vec.Doc2Vec(size=300, window=5, min_count=10)
model.build_vocab(doc2_vec_list)
model.train(doc2_vec_list, total_examples=model.corpus_count, epochs=500, start_alpha=0.25, end_alpha=0.025)
# </editor-fold>

# <editor-fold desc='plot some of the doc2vec things'>

A = model.docvecs.doctag_syn0
A_color = transform_to_color(range(len(model.docvecs.doctag_syn0)))
tsne_plot(A, A_color)

test_things = []
for i_doc in doc2_vec_list:
    test_things.append(model.infer_vector(i_doc.words))

B = test_things
B_color = transform_to_color(range(len(model.docvecs.doctag_syn0)))
# ... How do I know that these are plotted correctly?
tsne_plot(B, B_color)
# Maybe compare the label-vec.. check distance between the two..
# Make a metric where you can asses the result of the model....

# </editor-fold>

# <editor-fold desc='Run several algorithms'>

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)
classifier.score(test_arrays, test_labels)

from sklearn import svm
clf = svm.SVC()
clf.fit(train_arrays, train_labels)
clf.score(test_arrays, test_labels)

# </editor-fold>
