#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This scripts contains three classes that can help to analyse or convert pieces of text.

CleanText
 - remove stopwords from text
 - remove punctuation from text
 - remove integers from text
 - lemmatizes text (English only)
 - stems text (Dutch or English)


AnalyseDoc
 - get ngrams
 - most common ngrams
 - get Named Entities
 - most common ner
 - get summary

"""
import nltk
import collections
import string
import itertools
import gensim.summarization.keywords as kwrds_textRank
import numpy as np
from project.RegulatoryOffice.fileexplorer import KnowledgeGraph



class CleanText:
    """
    This class can clean your string by stemming or removing stopwords.
    Most NLTK things are in English.. and I wanted to create my own. So here it is.

    This class is build up by using string specific functions and list specific functions
    """
    def __init__(self, text, language='english'):
        """

        :param text: should be list as input
        :param language:
        """
        self.text = text

        self.language = language.lower()
        self.stopwords = set(nltk.corpus.stopwords.words(self.language) + list(string.punctuation))
        self._trans_punc = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        self._remove_digits = str.maketrans('', '', string.digits)
        self._lemmatizer = nltk.stem.WordNetLemmatizer()

        if self.language == 'dutch':
            self.stemmer = nltk.stem.snowball.DutchStemmer()
        elif self.language == 'english':
            self.stemmer = nltk.stem.snowball.PorterStemmer()

    def _lemmatize_string(self, text):
        """

        :param text: input string
        :return: lemmatized string
        """
        assert self.language == 'english'
        # This translation dict is used to convert pos-tags to pos-tags that can be used in the WordNet Lemmatizer.
        trans_dict = {'j': 'a', 'v': 'v', 'n': 'n', 'r': 'r'}

        text_pos = self._pos_tag_string(text)
        text_lem = [self._lemmatizer.lemmatize(x, pos=trans_dict.get(y[0].lower(), 'n')) for x, y in text_pos]
        return ' '.join(text_lem)

    def _pos_tag_string(self, text):
        """

        :param text: input string
        :return: postags of that string...
        """
        # nltk.help.upenn_tagset('V')
        # wordnet._FILEMAP
        #
        assert self.language == 'english'

        word = nltk.tokenize.word_tokenize(text)
        text_pos = nltk.tag.pos_tag(word)
        return text_pos

    def _remove_stopwords_string(self, text):
        """
        Removes stopwords from a splitted string, and then joining it agian.
        :param text: string as input
        :param return: string with words removed
        """
        word = nltk.tokenize.word_tokenize(text)
        string_filtered = [w for w in word if w.lower() not in self.stopwords]
        return ' '.join(string_filtered)

    def _stem_string(self, text):
        """
        Stems the given string by splitting it, and then joining it again.
        :param text: a string..
        :param return: another string, but stemmed..
        """
        word = nltk.tokenize.word_tokenize(text)
        string_stemmed = [self.stemmer.stem(w) for w in word]
        return ' '.join(string_stemmed)

    def _remove_punc_string(self, text):
        """
        Removes the punctuation in a string
        :param text:
        :return:
        """
        # return text.translate(self._trans_punc).lower()
        return text.translate(self._trans_punc)

    def _remove_int_string(self, text):
        """

        :param text: input string..
        :return: string without digits
        """
        return text.translate(self._remove_digits)

    def stem_text(self):
        """
        Stems the given articles
        """
        article_stemmed = [self._stem_string(a) for a in self.text]
        return CleanText(article_stemmed, language=self.language)

    def remove_stopwords_text(self):
        """
        Stems the given articles
        """
        article_filtered = [self._remove_stopwords_string(a) for a in self.text]
        return CleanText(article_filtered, language=self.language)

    def remove_punc_text(self):
        """
        Removes punctuation of the content of a list
        """
        article_no_punc = [self._remove_punc_string(x) for x in self.text]
        return CleanText(article_no_punc, language=self.language)

    def lemmatize_text(self):
        """
        Does lemmatization
        """
        article_lem = [self._lemmatize_string(x) for x in self.text]
        return CleanText(article_lem, language=self.language)

    def remove_int_text(self):
        """
        Removes all int
        :return:
        """
        article_no_int = [self._remove_int_string(x) for x in self.text]
        return CleanText(article_no_int, language=self.language)

class AnalyseDoc:
    """
    Used to analyse a piece of text...
    Input should be a list of text pieces
    """
    def __init__(self, text, language='english'):
        """
        :param text: List of sentences
        :type text: list
        :param language:
        """
        self.text = text
        self.language = language.lower()
        self.text_word = self._create_sent_word()

    def _create_sent_word(self):
        """
        needs to create the words per sentence...
        - text_sent_word will have lost the 'paragraph' information
        when you have provided a list of strings.
        - text_sent_word contains a list of list with words. Each element in the list is a sentence, and the list in
        that list contains all the words.
        :return:
        """
        text_word = ''
        text_word = [nltk.tokenize.word_tokenize(x, language=self.language) for x in self.text]
        return list(itertools.chain(*text_word))

    def get_ngram(self, n):
        """
        Returns the ngram of the input text..

        :param n:
        :param k:
        :return:
        """
        from helper.miscfunction import find_ngrams

        return list(find_ngrams(self.text_word, n))

    def most_common_ngrams(self, n, k):
        """
        Returns most k-common n-grams.

        You can perform n-gram calculations either sentence-bounded or sentence-unbouded with text_sent_word. When
        you are performing an unbounded test, then you first need to 'unpack' the list of list. This is done by
        choosing per_sentence=True or False.

        However, this might not be of such a great influence...
        Fun detail to keep in mind though

        :param n: size of ngram
        :param k: size of most common returned
        :return: top k n-grams
        """

        return collections.Counter(self.get_ngram(n)).most_common(k)

    def get_ner(self):
        """
        :param self:
        :return:
        """
        assert self.language == 'english'

        entities = []

        for i_text in self.text:
            chunks = nltk.ne_chunk(nltk.tag.pos_tag(nltk.tokenize.word_tokenize(i_text)))
            entities.extend([chunk for chunk in chunks if hasattr(chunk, 'label')])

        return entities

    def most_common_ner(self, n):
        """

        :param n:
        :return:
        """
        text_ner = self.get_ner()  # Works, but needs some post-processing
        text_ner_words = [' '.join([x[0] for x in y.leaves()]) for y in text_ner]
        text_ner_labels = [x.label() for x in text_ner]
        text_ner_combi = zip(text_ner_words, text_ner_labels)
        return collections.Counter(text_ner_combi).most_common(n)

    def get_summary(self, n=3, title='This is a fake title'):
        """
        Obtaining a summary... The text deliver should be a full string, not a list.

        :param n: amount of sentences
        :param title: need not to be empty.. otherwise thing crashes
        :return:
        """
        from summarizer import summarize as githubsummarize

        sum_text = githubsummarize(title, ' '.join(self.text), n)  # As input: full text
        return sum_text

    def most_common_keywords(self, n):
        """
        Returns the keywords.. that are mos common in the provided text
        :param n:
        :return:
        """
        output_keywords = kwrds_textRank(' '.join(self.text), words=n, split=True, scores=True,
                                                 pos_filter=None, lemmatize=True, deacc=True)
        return output_keywords

    #     """
    #     I want to change this one
    #     THis does not add anything
    #
    #     Split a piece of text based on another piece of text
    #     :param self:
    #     :param n:
    #     :param m: the result of the summary per paragraph
    #     :param title:
    #     :return:
    #     """
    #     fs = FrequencySummarizer(language=self.language)
    #     method_1 = ''
    #     method_2 = ''
    #     if isinstance(self.text, str):
    #         method_1 = fs.summarize(self.text, n)  # self created
    #         method_2 = summarizer.summarize(title, self.text, n)  # use one from a package
    #     elif isinstance(self.text, list):
    #         sum_text = self.text
    #         paragraph_sum_1 = [fs.summarize(x, m) for x in sum_text if len(x) > 5]
    #         paragraph_sum_1 = ' '.join([x for b in paragraph_sum_1 for x in b])
    #         method_1 = fs.summarize(paragraph_sum_1, n)
    #
    #         paragraph_sum_2 = [summarizer.summarize(title, x, m) for x in self.text if len(x) > 0]
    #         paragraph_sum_2 = ' '.join([x for b in paragraph_sum_2 for x in b])
    #         method_2 = summarizer.summarize(title, paragraph_sum_2, n)  # use one from a package
    #
    #     return method_1, method_2

    #
    # def get_topics_lda(self, num_topics=3, passes=10):
    #     """
    #     aa
    #     :param self:
    #     :param num_topics:
    #     :param passes:
    #     :return:
    #     """
    #     tfidf = ''
    #     if isinstance(self.text, str):
    #         tfidf = TFIDF(self.text_sent)
    #     elif isinstance(self.text, list):
    #         tfidf = TFIDF(self.text)
    #
    #     lda_corpus = []
    #     This will loop over the 'documents'
    #     Or in our case.. sentences or paragraphs
        # for i_count in tfidf.TF_count:
        #     index_words = np.nonzero(i_count)  # Find the words in document i_count
        #     value_words = i_count[index_words].astype(int)  # Retrieve the non zero words
        #     lda_corpus.append(list(zip(index_words[0], value_words)))  # Combine the info
        #
        # lda_dictionary = tfidf.vocab_dict
        # lda_dictionary = {v: k for k, v in lda_dictionary.items()}
        # frequency pair.
        # lda_model = gensim.models.ldamodel.LdaModel(lda_corpus, num_topics=num_topics, id2word=lda_dictionary,
        #                                             passes=passes)
        # return lda_model
    #
    # def _sent_to_svd(self):
    #     """
    #
    #     :param self:
    #     :return:
    #     """
    #     tfidf = ''
    #     if isinstance(self.text, str):
    #         tfidf = TFIDF(self.text_sent)
    #     elif isinstance(self.text, list):
    #         tfidf = TFIDF(self.text)
    #
    #     tf_count_list = tfidf.TF_count
    #     term_sent = pd.DataFrame(tf_count_list)
    #     u, d, v = np.linalg.svd(term_sent.T, full_matrices=False)
    #     return u, d, v, tfidf.vocab_dict
    #
    # def get_topics_psa(self):
    #     """
    #     Split a piece of text based on another piece of text
    #     :return:
    #     """
    #     u_sum, _, _, dict_sum = self._sent_to_svd()
    #     pca_vec = u_sum[:, 0]
    #     index_nonzero = np.where(~np.isclose(pca_vec, 0))
    #     items = list(dict_sum.keys())
    #     result = [(round(pca_vec[x], 3), items[x]) for x in index_nonzero[0]]
    #     return result

class MultiAnalyseDoc(AnalyseDoc):

    def __init__(self, list_text, language='english'):
        """

        :param list_text: List of sentences... Hence [[sent_1a, sent_2a,...], [sent1b, sent2b,...]]
        :param language:
        """
        self.language = language.lower()
        self.AnalyseDoc_list = [AnalyseDoc(x, language) for x in list_text]

    def _score_on_tfidf(self, input_list, k):
        """
        Score list of strings based on tfidf..

        :param input_list: list of sentences
        :param k: top k amount of words are being taken
        :return:
        """
        from embedding.docembedding import TFIDF

        A = TFIDF(input_list)
        tf_idf_ngram = A.TF_count*A.IDF
        res_conc = np.concatenate(tf_idf_ngram)
        res_id = np.concatenate([list(range(A.word_N))] * A.idf_N)
        res_best_id = np.argsort(-res_conc)
        # res_conc[res_best_id[0:5]]  # This shows me the score...
        derp = res_id[res_best_id[:k]]  # This shows me the id for the vocab
        return [A.vocab_dict_num[x] for x in derp]

    def multi_ngram(self, n, k):
        """

        :param n:
        :return:

        """
        # Here we concat all the ngrams...
        ngram_string_doc = []
        for x in self.AnalyseDoc_list:
            temp = x.most_common_ngrams(n, k)
            ngram_string = ' '.join(['_'.join(x[0]) for x in temp])
            ngram_string_doc.append(ngram_string)

        res = self._score_on_tfidf(ngram_string_doc, k)
        res = [x.split('_') for x in res]

        return  res # This shows me the real words..

    def multi_ner(self, k):
        """

        :param k:
        :return:

        """
        # Here we concat all the ngrams...
        ner_string_doc = []
        for x in self.AnalyseDoc_list:
            temp = x.most_common_ner(k)
            ner_string = ' '.join(['_'.join(x[0]) for x in temp])
            ner_string_doc.append(ner_string)

        res = self._score_on_tfidf(ner_string_doc, k)
        res = [x.split('_') for x in res]

        return  res # This shows me the real words..

    def multi_keyword(self, k):
        """
        Return the ensemble of keywords from multiple documents

        :param k:
        :return:

        """
        # Here we concat all the ngrams...
        keyword_string_doc = []
        for x in self.AnalyseDoc_list:
            temp = x.most_common_keywords(k)
            keyword_string = ' '.join([x[0] for x in temp])
            keyword_string_doc.append(keyword_string)

        res = self._score_on_tfidf(keyword_string_doc, k)
        # res = [x.split('_') for x in res]

        return  res # This shows me the real words..

class EurLexMultiAnalyseDoc(KnowledgeGraph):

    def __init__(self, db_src_path, list_file_name):
        """
        Dingen
        """
        self.list_file_name = list_file_name
        super(EurLexMultiAnalyseDoc, self).__init__(db_src_path)

    def _score_on_tfidf(self, input_list, k):
        """
        Score list of strings based on tfidf..

        :param input_list: list of sentences
        :param k: top k amount of words are being taken
        :return:
        """
        from embedding.docembedding import TFIDF

        A = TFIDF(input_list)
        tf_idf_ngram = A.TF_count*A.IDF
        res_conc = np.concatenate(tf_idf_ngram)
        res_id = np.concatenate([list(range(A.word_N))] * A.idf_N)
        res_best_id = np.argsort(-res_conc)
        # res_conc[res_best_id[0:5]]  # This shows me the score...
        derp = res_id[res_best_id[:k]]  # This shows me the id for the vocab
        return [A.vocab_dict_num[x] for x in derp]

    def multi_ngram(self, k):
        """
        Results are based on the predefined json dump...

        :param n:
        :return:

        """
        # Here we concat all the ngrams...
        ngram_string_doc = []
        for x in self.list_file_name:
            temp = self.db_json[x]['Trigram'][:k]
            ngram_string = ' '.join([x.replace(' ','_') for x in temp])
            ngram_string_doc.append(ngram_string)

        res = self._score_on_tfidf(ngram_string_doc, k)
        # res = [x.split('_') for x in res]
        res = [x.replace('_', ' ') for x in res]

        return  res # This shows me the real words..

    def multi_ner(self, k):
        """

        :param k:
        :return:

        """
        # Here we concat all the ngrams...
        ner_string_doc = []
        for x in self.list_file_name:
            temp = self.db_json[x]['NER'][:k]
            ner_string = ' '.join([x.replace(' ','_') for x in temp])
            ner_string_doc.append(ner_string)

        res = self._score_on_tfidf(ner_string_doc, k)
        # res = [x.split('_') for x in res]

        return  res # This shows me the real words..

    def multi_keyword(self, k):
        """
        Return the ensemble of keywords from multiple documents

        :param k:
        :return:

        """
        # Here we concat all the ngrams...
        keyword_string_doc = []
        for x in self.list_file_name:
            temp = self.db_json[x]['Keywords'][:k]
            keyword_string = ' '.join([x for x in temp])
            keyword_string_doc.append(keyword_string)

        res = self._score_on_tfidf(keyword_string_doc, k)
        res = [x.replace('_', ' ') for x in res]

        return  res # This shows me the real words..

# TODO check if using the multi document thing actually helps to reduce words like Article...