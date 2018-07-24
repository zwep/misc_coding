# encoding: utf-8

"""
This should be turned into a class of multiple assesments

Here we have some example of a summarizer
THIS IS HOW GENISM NEEDS THE MODELS
This is a document, but instead of a list of words, it is a list of tuples where each tuple is a word id and
Check out http://christop.club/2014/05/06/using-gensim-for-lda/

explanation LSA LDA
https://cs.stanford.edu/~ppasupat/a9online/1140.html
https://www.slideshare.net/ananth/word-representation-svd-lsa-word2vec

"""


import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize

from summarization.frequencysummarizer import FrequencySummarizer
from embedding.docembedding import TFIDF


def sent_to_svd(sent):
    """
    Convert a list of sentences
    :param sent:
    :return:
    """
    tf = TFIDF(sent)
    tf_count_list = tf.TF_count
    term_sent = pd.DataFrame(tf_count_list)
    u, d, v = np.linalg.svd(term_sent.T, full_matrices=False)
    return u, d, v, tf.vocab_dict


def compare_summary(article_text, n):
    """

    :param article_text:
    :param n:
    :return:
    """

    article_sentence = sent_tokenize(article_text)
    fs = FrequencySummarizer()
    summary_sentence = fs.summarize(article_text, n)

    u_sum, d_sum, v_sum, dict_sum = sent_to_svd(summary_sentence)
    u_art, d_art, v_art, dict_art = sent_to_svd(article_sentence)

    # Transform the summary matrix to the big matrix
    b = [(dict_art[x], dict_sum[x]) for x in list(dict_sum.keys())]
    a_index = pd.DataFrame(b, columns=['orig_index', 'new_index']).sort_values('new_index')

    i_index = a_index['orig_index'].values
    u_art_sel = u_art[i_index, :]

    # LSA metric for comparing topics...
    cos_phi = np.dot(u_art_sel[:, 0], u_sum[:, 0])

    return cos_phi, dict_sum, u_art_sel, u_sum


def get_topics_pca(pca_vec, vocab_dict):
    """

    :param pca_vec:
    :param vocab_dict:
    :return:
    """
    index_nonzero = np.where(~np.isclose(pca_vec, 0))
    items = list(vocab_dict.keys())
    result = [str(round(pca_vec[x], 3)) + "*" + items[x] for x in index_nonzero[0]]
    return '+ '.join(result)


to_summarize = 'NEW DELHI: The Dalai Lama has spoken out for the first time about the Rohingya refugee crisis, ' \
               'saying Buddha would have helped Muslims fleeing violence in Buddhist-majority Myanmar. Hundreds of ' \
               'thousands of Rohingya have arrived in Bangladesh in recent weeks after violence flared in ' \
               'neighbouring Myanmar, where the stateless Muslim minority has endured decades of persecution. The top' \
               'Buddhist ' \
               'leader is the latest Nobel peace laureate to speak out against the violence, which the UN special ' \
               'rapporteur on human rights in Myanmar says may have killed more than 1,000 people, most of them ' \
               'Rohingya. "Those people who are sort of harassing some Muslims, they should remember Buddha,' \
               '" the Dalai Lama told journalists who asked him about the crisis on Friday (Sep 8) evening. "He would \
               definitely give help to those poor Muslims. So still I feel that. So very sad." Myanmars population ' \
               'is overwhelmingly Buddhist and there is widespread hatred for the Rohingya, who are denied ' \
               'citizenship and labelled illegal "Bengali" immigrants. Buddhist nationalists, led by firebrand monks, \
               have operated a long Islamophobic campaign calling for them to be pushed out of the country. Myanmars ' \
               'de facto civilian leader Aung San Suu Kyi has been condemned for her refusal to intervene in support ' \
               'of the Rohingya, including by fellow Nobel laureates Malala Yousafzai and Desmond Tutu. Archbishop ' \
               'Tutu, who became the moral voice of South Africa after helping dismantle apartheid there, ' \
               'last week urged her to speak out. "If the political price of your ascension to the highest office in ' \
               'Myanmar is your silence, the price is surely too steep," Tutu said in a statement.'


# Whyy dont you train the words on a skipgram model as well
# Finalize the topic selection and summarizaiton.... into a class
# EIther merge with existintg thing or createa new one
# get topics with LDA...

fs = FrequencySummarizer()
summary_sentence = fs.summarize(to_summarize, 4)

cos_phi_thing, dict_words, svd_art, svd_sum = compare_summary(to_summarize, 4)

# topicslda = get_topics_lda(to_summarize)
# topicsldasum = get_topics_lda('. '.join(summary_sentence))

get_topics_pca(svd_art[:, 0], dict_words)
get_topics_pca(svd_sum[:, 0], dict_words)
