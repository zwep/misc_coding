# -*- coding: utf-8 -*-

"""
This file contains only the FrequencySummarizer
https://glowingpython.blogspot.nl/2014/09/text-summarization-with-nltk.html
"""

import nltk
import heapq
import collections
import string


class FrequencySummarizer:
    """
    Self created frequency summarizer
    """
    def __init__(self, min_cut=0.1, max_cut=0.9, language='dutch'):
        """
        Initilize the text summarizer.
        Words that have a frequency term lower than min_cut
        or higer than max_cut will be ignored.
        """
        self._min_cut = min_cut
        self._max_cut = max_cut
        self._stopwords = set(nltk.corpus.stopwords.words(language) + list(string.punctuation))
        self._freq = float()

    def _compute_frequencies(self, word_sent):
        """
        Compute the frequency of each of word.
        Input:
        word_sent, a list of sentences already tokenized.
        Output:
        freq, a dictionary where freq[w] is the frequency of w.
        """
        freq = collections.defaultdict(int)
        for s in word_sent:
            for word in s:
                if word not in self._stopwords:
                    freq[word] += 1
        # frequencies normalization and filtering
        m = float(max(freq.values()))
        for w in list(freq):
            freq[w] /= m
            if freq[w] >= self._max_cut or freq[w] <= self._min_cut:
                del freq[w]
        return freq

    def summarize(self, text, n):
        """
        Return a list of n sentences
        which represent the summary of text.
        """
        if isinstance(text, str):
            sents = nltk.tokenize.sent_tokenize(text)
        elif isinstance(text, list):
            sents = text
        else:
            print('You have supplied text that is of type ', type(text))
            return text

        assert n <= len(sents)

        word_sent = [nltk.tokenize.word_tokenize(s.lower()) for s in sents]
        self._freq = self._compute_frequencies(word_sent)
        ranking = collections.defaultdict(int)

        for i, sent in enumerate(word_sent):
            for w in sent:
                if w in self._freq:
                    ranking[i] += self._freq[w]

        sents_idx = self._rank(ranking, n)

        return [sents[j] for j in sents_idx]

    @staticmethod
    def _rank(ranking, n):
        """ return the first n sentences with highest ranking """
        return heapq.nlargest(n, ranking, key=ranking.get)
