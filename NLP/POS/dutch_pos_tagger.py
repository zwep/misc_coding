# encoding: utf-8

"""
http://text-processing.com/demo/tag/
https://github.com/evanmiltenburg/Dutch-tagger

We got a POS tagger and a CHUNK tagger. In combination they can be used to apply NER...


"""

import os
import nltk

from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import alpino as alp  # Trained on ALP data.


tagger = PerceptronTagger(load=False)
os.chdir(r'D:\nlp_lib')
tagger.load('model.perc.dutch_tagger_small.pickle')  # I don't know the source of training data

# Tag a sentence.
tagger.tag('Alle vogels zijn nesten begonnen , behalve ik en jij .'.split())

training_corpus = alp.tagged_sents()
unitagger = nltk.tag.UnigramTagger(training_corpus)
bitagger = nltk.tag.BigramTagger(training_corpus, backoff=unitagger)
perctagger = PerceptronTagger(load=True)  # What does load=True mean??
perctagger.train(training_corpus)

sent = 'NLTK is een goeda taal voor NLP'.split()
bitagger.tag(sent)
unitagger.tag(sent)
perctagger.tag(sent)