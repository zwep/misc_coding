"""
Topic modelling on news based on LDA
"""

from helper.loadrss import *
from helper.miscfunction import *
from textprocessing.processtext import *

import pandas as pd
"""
Function we are going to build/use
"""

def sent_to_svd(sent):
    tf = TFIDF(sent)
    tf_count_list = tf.TF_count
    term_sent = pd.DataFrame(tf_count_list)
    u, d, v = np.linalg.svd(term_sent.T, full_matrices=False)
    return u, d, v, tf.vocab_dict

def get_topics_pca(pca_vec, vocab_dict):
    index_nonzero = np.where(~np.isclose(pca_vec, 0))
    items = list(vocab_dict.keys())
    result = [str(round(pca_vec[x], 3)) + "*" + items[x] for x in index_nonzero[0]]
    return '+ '.join(result)

"""
prepping LSA
"""

article, article_label = get_url_content()
article_label_color = transform_to_color(article_label)
article_clean = CleanText(article)
article_clean_1 = article_clean.remove_punc_text().remove_stopwords_text().text
article_clean_2 = article_clean.remove_stopwords_text().text

long_article = [x for x in article if len(x) > 2000]
long_article_2 = [x for x in article_clean_2 if len(x) > 2000]
long_article_2[3]

# Now maybe clean the data...
sent_long_article = [nltk.tokenize.sent_tokenize(x) for x in long_article]

i = 8
derp = CleanText(sent_long_article[i]).remove_punc_text().remove_stopwords_text().text
u_sum, d_sum, v_sum, dict_sum = sent_to_svd(derp)
get_topics_pca(u_sum[:, 0], dict_sum)



def extract_entities(text):
    entities = []
    for sentence in sent_tokenize(text):
        chunks = ne_chunk(pos_tag(word_tokenize(sentence)))
        entities.extend([chunk for chunk in chunks if hasattr(chunk, 'label')])
    return entities