"""
Topic modelling on news based on LDA
"""
import gensim.models

from helper.loadrss import *
from helper.miscfunction import *
from textprocessing.processtext import *
from embedding.docembedding import *

"""
Function we are going to build/use
"""


def get_topics_lda(text):
    """

    :param text: full text (string) article
    :return: lda model
    """
    article_sent = nltk.tokenize.sent_tokenize(text)
    tfidf = TFIDF(article_sent)
    lda_corpus = []
    for i_count in tfidf.TF_count:
        z = np.nonzero(i_count)
        derp = i_count[z].astype(int)
        lda_corpus.append(list(zip(z[0], derp)))

    ""
    lda_dictionary = tfidf.vocab_dict
    lda_dictionary = {v: k for k, v in lda_dictionary.items()}
    # frequency pair.
    ldamodel = gensim.models.ldamodel.LdaModel(lda_corpus, num_topics=3, id2word=lda_dictionary, passes=10)
    print(ldamodel.print_topics())
    return ldamodel


"""
prepping LDA
"""

article, article_label = get_url_content()
article_label_color = transform_to_color(article_label)
article_clean = CleanText(article)
article_clean_1 = article_clean.remove_punc_text().remove_stopwords_text().text
article_clean_2 = article_clean.remove_stopwords_text().text

long_article = [x for x in article if len(x) > 2000]
long_article_2 = [x for x in article_clean_2 if len(x) > 2000]
long_article_2[3]
get_topics_lda(long_article_2[0])
get_topics_lda(long_article[0])
get_topics_lda(long_article[5])


# uitleg LSA LDA
# https://cs.stanford.edu/~ppasupat/a9online/1140.html
# https://www.slideshare.net/ananth/word-representation-svd-lsa-word2vec
# frequency pair.
