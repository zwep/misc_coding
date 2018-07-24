# encoding: utf-8

"""
Check amount of RAM

"""

import os

import gensim.models
from psutil import virtual_memory

from helper.loadrss import RssUrl
from helper.miscfunction import *
from textprocessing.processtext import *

mem = virtual_memory()
# Check amount of mem availalbe
print(mem.total / 10 ** 9)


if __name__ == '__main__':
    A = RssUrl()
    article, article_label = A.get_all_content()
    article_label_color = transform_to_color(article_label)

    article_label_simplified = np.array(article_label)
    article_label_simplified[article_label_simplified == 'nossportalgemeent'] = 'Sport'
    article_label_simplified[article_label_simplified == 'nosnieuwsbinnenland'] = 'nederland'
    article_label_simplified_color = transform_to_color(article_label)

    article_clean = CleanText(article)
    article_clean = article_clean.remove_punc_text().remove_stopwords_text().text

    path_word2vec_emb = r'D:\data\text\Wordembed'
    os.chdir(path_word2vec_emb)
    model = gensim.models.KeyedVectors.load_word2vec_format('wikipedia-320.txt', binary=False, limit=500000)

    # initialize TFIDF class...
    # Used to obtain the nBOW for the given documents
    p = TFIDF(article_clean)
    article_vocab = p.vocab_document
    N = p.idf_N

    # Check which words are in the model and which are in the documents
    model_to_vocab = list(map(model.vocab.get, article_vocab))
    model_subset_index = np.nonzero(model_to_vocab)
    model_subset_word = article_vocab[model_subset_index]
    # Don't take the realisation of all the word-embedding
    # But only of those that are actually found in the document
    model_subset = np.array([model[i_word] for i_word in model_subset_word])
    X = model_subset.transpose()

    document_n_bow = np.array(p.tf_freq())
    document_n_bow_subset = document_n_bow[:, model_subset_index[0]]
    D = document_n_bow_subset.transpose()

    # These XDs are thus an weighted average of the word embedding per document
    XD_list = [np.matmul(X, D[:, i]) for i in range(N)]
    WCD_value = [np.linalg.norm(XD_list[i] - XD_list[j]) for i in range(N) for j in np.arange(N)]
    WCD_index = [(i, j) for i in range(N) for j in np.arange(N)]

    # This shows the WMD between all the documents.. pretty funny format
    WCD = np.ndarray(shape=(N, N))
    for i, index in enumerate(WCD_index):
        WCD[index] = WCD_value[i]

    """
    Score the articles with the XD features
    """

    print(score_logistic_regression(np.array(XD_list), np.array(article_label), model_subset_word))
    print(score_logistic_regression(np.array(XD_list), np.array(article_label_simplified), model_subset_word))
    tsne_plot(XD_list, article_label_color)
    tsne_plot(XD_list, article_label_simplified_color)

    """
    Trying out the scoring with the frequencies approach..
    """

    # Selection of something.. not sure what
    i_model = model_subset_index[0]
    # Trying out TF
    print(score_logistic_regression(p.tf_freq()[:, i_model], np.array(article_label), model_subset_word))
    print(score_logistic_regression(p.tf_freq()[:, i_model], np.array(article_label_simplified), model_subset_word))
    print(score_logistic_regression(p.tf_bin()[:, i_model], np.array(article_label), model_subset_word))
    print(score_logistic_regression(p.tf_bin()[:, i_model], np.array(article_label_simplified), model_subset_word))
    print(score_logistic_regression(p.tf_augmented_freq()[:, i_model], np.array(article_label),
                                    model_subset_word))
    print(score_logistic_regression(p.tf_log_freq()[:, i_model], np.array(article_label), model_subset_word))
    # Trying out TFIDF
    print(score_logistic_regression((p.tf_freq()*p.IDF())[:, i_model], np.array(article_label),
                                    model_subset_word))
    print(score_logistic_regression(p.idf_max()[:, i_model], np.array(article_label), model_subset_word))
    print(score_logistic_regression(p.idf_prob()[:, i_model], np.array(article_label), model_subset_word))
    print(score_logistic_regression(p.idf_smooth()[:, i_model], np.array(article_label), model_subset_word))
    # Trying out XD
    print(score_logistic_regression(XD_list, np.array(article_label), model_subset_word))

    # Plotting with tsne
    tsne_plot(p.tf_freq(), article_label_color)
    tsne_plot(p.tf_bin(), article_label_color)
    tsne_plot(p.tf_bin(), article_label_simplified_color)
