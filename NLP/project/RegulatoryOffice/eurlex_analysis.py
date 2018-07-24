#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
In this document we use some machine learning techniques on the EURLEX data to see how that works out.


# TODO Word distance
# TODO Use Word2Vec to create embedded cententeses, or use FastText to get subword info
# TODO Analyse the HTML and what is in there, and see if the font size/markup can be helpfull
# TODO learn how doc2vec can be used WITHOUT the trained word vectors. Because that is weird
# TODO doc2vec distribution compared to the WMD diistirbution of paragraphs
# TODO kan ik de drie documenten van elkaar onderscheiden??

Notes on script

Used resource
    Doc2Vec
    https://radimrehurek.com/gensim/models/doc2vec.html
    https://medium.com/@klintcho/doc2vec-tutorial-using-gensim-ab3ac03d3a1
    https://www.kaggle.com/c/word2vec-nlp-tutorial/discussion/12287
    http://linanqiu.github.io/2015/10/07/word2vec-sentiment/
    https://rare-technologies.com/doc2vec-tutorial/

Here you can see the inner workings of the summarize.summarizer package
    https://github.com/michigan-com/summarizer/blob/master/summarizer/summarizer.py
"""

import gensim
import numpy as np
import re

from gensim.models.doc2vec import TaggedDocument

from textprocessing.processtext import *
from summarization.frequencysummarizer import FrequencySummarizer

from helper.miscfunction import transform_to_color
from helper.plotfunction import tsne_plot

from embedding.docembedding import TFIDF

import project.RegulatoryOffice.fileexplorer as reg_fe
import project.RegulatoryOffice.ridfunctions as proc_rid

"""
Define functions
"""


def build_sent_doc2vec(input_list_doc):
    """
    Maybe in a later stage we can put this in a class and in the /embedding thnig
    :param input_list_doc:
    :return:
    """
    return [TaggedDocument(words=nltk.tokenize.word_tokenize(i_doc), tags=['SECT '+str(i)]) for i, i_doc in enumerate(
        input_list_doc)]


"""
Loading and analysing the documents
"""


dir_data = r'D:\temp'
dir_reg_doc = dir_data + r'\reg_doc'
dir_reg_html = dir_reg_doc + r'\html'
name_data = 'Export_RID_16012018.xlsx'
n_doc = 2

rid_data = proc_rid.get_RID(name_data, dir_reg_doc)  # Gives back a dictionary
html_obj = proc_rid.get_html_obj(dir_reg_html, n=n_doc)  # Gives back a tuple of (content, name)

for i_obj in html_obj:
    A = reg_fe.ExtractRegDoc(*i_obj, rid_data=rid_data)


# --- preprocessing ---
# B = CleanText(list_text, language='english')
B = CleanText(sub_list_text_sent, language='english')
B = CleanText(sub_concat_text_p, language='english')

B.stopwords.add('—')
B.stopwords.add('___________')
B.stopwords.add('’')
B.stopwords.add('‘')
B.stopwords.add('“')
B.stopwords.add('”')

B_stop = B.remove_stopwords_text()
B_lem_stop = B.remove_stopwords_text().lemmatize_text()
B_lem_stop_int_punc = B.remove_stopwords_text().lemmatize_text().remove_int_text().remove_punc_text()

# --- counting ---
A = AnalyseDoc(B_stop.text)
# A = AnalyseDoc(B_lem_stop.text)
A.most_common_ngrams(20, 1)
A.most_common_ngrams(10, 2)
for x in A.most_common_ngrams(5, 3):
    print("\"" + ' '.join(x[0]) + "\"" + '\t' + str(x[1]))

A_most_ner = A.most_common_ner(10)
for x in A_most_ner:
    print("Entity:\t\"" + '\"\tCategory:\t'.join(x[0]) + '\tCount:\t' + str(x[1]))

# --- LDA ---


def get_topics_lda(text, n):
    """

    :param text: list of text
    :param n: amount of topics...
    :return: lda model
    """
    tfidf = TFIDF(text)
    lda_corpus = []
    for i_count in tfidf.TF_count:
        z = np.nonzero(i_count)
        derp = i_count[z].astype(int)
        lda_corpus.append(list(zip(z[0], derp)))

    ""
    lda_dictionary = tfidf.vocab_dict
    lda_dictionary = {v: k for k, v in lda_dictionary.items()}
    # frequency pair.
    ldamodel = gensim.models.ldamodel.LdaModel(lda_corpus, num_topics=n, id2word=lda_dictionary, passes=20)
    print(ldamodel.print_topics())
    return ldamodel, lda_corpus


text_lda = get_topics_lda(nltk.tokenize.sent_tokenize(full_text), 10)
text_lda, lda_corp = get_topics_lda(B_lem_stop_int_punc.text, 15)
topic_lda = [text_lda.show_topic(i) for i in range(10)]
for i in topic_lda:
    print([j[0] for j in i])


# --- summarizations ---
# This also uses some TFIDF frequency, but also some scoring based on other content.
output_sum1 = githibsummarize('NWO', full_text, 3)  # As input: full text
output_sum2 = githibsummarize('ABN AMRO', ' '.join(B.text), 4)  # As input: full text

# This uses some simple ranking based on TFIDF frequency
test = FrequencySummarizer(language='english')
test.summarize(full_text, 3)
test.summarize(B_lem_stop.text, 3)
test.summarize(B.text, 3)

# This uses TextRank (a variant of pagerank)
gensim.summarization.summarize(full_text, ratio=0.05)
gensim.summarization.summarize(' '.join(B.text), ratio=0.2)
test = gensim.summarization.summarize(' '.join(sub_concat_text_p), ratio=0.05)


# --- wordembedding ---

# Using TextRank to get keywords
result_keywords = gensim.summarization.keywords(full_text, words=10, split=True, scores=True,
                                                pos_filter=None, lemmatize=True, deacc=True)

for i in result_keywords:
    print("\"" + i[0] + '\" \t' + '{:.3f}'.format(i[1][0]))

dum = full_text

for i in result_keywords:
    # dum = dum.replace(i[0], '')
    print(re.findall(i[0], dum)[0:10])

result_keywords = gensim.summarization.keywords(dum, words=10, split=True, scores=True,
                                                pos_filter=None, lemmatize=True, deacc=True)

# Using Google News word vectors - this can take some time
filename = '\GoogleNews-vectors-negative300.bin'
dirname = r'D:\data\text\Wordembed'
model_googlenews = gensim.models.KeyedVectors.load_word2vec_format(dirname + filename, binary=True)

# Using Doc2Vec to classify the document
dir_data = r'D:\temp\reg_doc\text'
import glob
import os
file_list = [os.path.basename(x) for x in glob.glob(dir_data + '\*.txt')]

# file_list = ['PSD2_example3.txt', 'GDPR_example2.txt', 'CRR2_example1.txt']
doc2_vec_list = []
googlenews_list = []

for i_file in file_list:
    print(i_file + '-----------')
    chosen_file = i_file  # file_list[0]
    full_text, list_text = get_text_data(chosen_file, dir_data)
    split_text_p = split_list_by_p([''] + list_text)
    concat_text_p = compact_list_of_lists(split_text_p)
    sub_concat_text_p = [x for x in concat_text_p if len(x) > 100]
    build_sent_p = build_sent_doc2vec(sub_concat_text_p)  # either use the sub concat text

    # build_sent_p = build_sent_doc2vec([full_text])  # or this full text...

    model = gensim.models.doc2vec.Doc2Vec(size=300, window=5, min_count=10)
    model.build_vocab(build_sent_p)
    model.train(build_sent_p, total_examples=model.corpus_count, epochs=20, start_alpha=0.25, end_alpha=0.025)
    doc2_vec_list.append(model.docvecs.doctag_syn0)

    # Wanted to check how the mean of the GoogleNews-word2vec model is compared to the doc2vec model. After some
    # simple runs, it turns out that doc2vec works better.
    # test = [np.mean([model_googlenews[x_word] for x_word in nltk.tokenize.word_tokenize(y) if x_word in
#                      model_googlenews.vocab], axis=0) for y in sub_concat_text_p]
#     test2 = np.concatenate([np.reshape(x, (1, len(x))) for x in test], axis=0)
#     googlenews_list.append(test2)


# Plot multiple documents
A = np.concatenate(doc2_vec_list)
A_color = transform_to_color(range(len(doc2_vec_list)))
N_color = [x.shape[0] for x in doc2_vec_list]
B_color = list(itertools.chain(*[[x] * N_color[i] for i, x in enumerate(A_color)]))
tsne_plot(A, B_color)

A = np.concatenate(googlenews_list)
A_color = transform_to_color(range(len(googlenews_list)))
N_color = [x.shape[0] for x in googlenews_list]
B_color = list(itertools.chain(*[[x] * N_color[i] for i, x in enumerate(A_color)]))
tsne_plot(A, B_color)


B = model.wv.syn0
B_color = transform_to_color(range(len(B)))
tsne_plot(B, B_color)

# dm = 1 -> PV-DM
# dm = 0 -> PV-DBOW
# size: dimension of feature vector
# window: max distance between the predicted word and context word used for prediction within a document
# alpha: initial learning rate
# min_count: ignore all words with total frequency lower than this
# max_vocab_size: limit RAM. Every 10 mil words is about 1 GB of RAM
# sample: used for downsampling
# iter: default from Word2Vec is 5... but values 10-20 are also normal
# db_words: trains the word vectors as well


"""
Analyse document
"""

test = ['(23)  This Directive should not apply to payment transactions made in cash since a single payments market '
        'for cash already exists. Nor should this Directive apply to payment transactions based on paper cheques '
        'since, by their nature, paper cheques cannot be processed as efficiently as other means of payment.',
        'Good practice in that area should, however, be based on the principles set out in this Directive.'
        '(24)  It is necessary to specify the categories of payment service providers which may legitimately provide '
        'payment services throughout the Union, namely, credit institutions which take deposits from users that can '
        'be used to fund payment transactions and which should continue to be subject to the prudential requirements '
        'laid down in Directive 2013/36/EU of the European Parliament and of the Council (1), electronic money '
        'institutions which issue electronic money that can be used to fund payment transactions and which should '
        'continue to be subject to the prudential requirements laid down in Directive 2009/110/EC, '
        'payment institutions and post office giro institutions which are so entitled under national law.',
        'The application of that legal framework should be confined to service providers who provide payment services '
        'as a regular occupation or business activity in accordance with this Directive.'
        '(25)  This Directive lays down rules on the execution of payment transactions where the funds are electronic '
        'money as defined in Directive 2009/110/EC.'
        'This Directive does not, however, regulate the issuance of electronic money as provided for in Directive '
        '2009/110/EC.'
        'Therefore, payment institutions should not be allowed to issue electronic money.']


test2 = ['DERP DERP (23)  This Directive should not apply to payment transactions made in cash since a single '
         'payments '
         'market '
        'for cash already exists. Nor should this Directive apply to payment transactions based on paper cheques '
        'since, by their nature, paper cheques cannot be processed as efficiently as other means of payment.',
        'Good practice in that DERP area should, however, be based on the principles set out in this Directive.'
        '(24)  It is necessary to specify the categories of payment service providers which may legitimately provide '
        'payment services throughout the Union, namely, credit institutions which take deposits from users that can '
        'be used to fund payment transactions and which should continue to be subject to the prudential requirements '
        'laid down in Directive 2013/36/EU of the European Parliament and of the Council (1), electronic money '
        'institutionDERPEDERPs which issue electronic money that can be used to fund payment transactions and which '
        'should '
        'continue to be subject to the prudential requirements laid down in Directive 2009/110/EC, '
        'payment institutions and post office giro i DERP DERP nstitutions which are so entitled under national law.',
        'The application of that legal framework should be confined to service providers who provide payment services '
        'as a regular occupation or business activity in accordance with this Directive.'
        '(25)  This Directive lays down rules on the execution of payment transactions where the funds are electronic '
        'money as defined in Directive 2009/110/EC.'
        'This Directive does not, however, regulate the issuance of electronic money as provided for in Directive '
        '2009/110/EC.'
        'Therefore, payment in DERP DERP stitutions should not be allowed to issue electronic money.']

list_test = [test, test2]
