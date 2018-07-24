# encoding: utf-8

"""

In this file we are going to explore the wonders of StandordNLP packages and their dependency parsers for the fun of it.

This needs a bit more work


# In order to start the CoreNLP server... follow all instructions mentioned here
#
# Then move to your Powershell (or cmd) and execute the following like
#
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
#
# Don't worry that you don't see edu.stanford.nlp.pipeline.StanfordCoreNLPServer in the directory..
#
# timeout is set in miliseconds... so the server will be up for quite a short time now.
# Now we have set up the server at http://localhost:9000

"""

import re
from nltk.parse.stanford import StanfordDependencyParser

# path_to_jar = 'path_to/stanford-parser-full-2014-08-27/stanford-parser.jar'
path_to_jar = r'D:\nlp_lib\stanford-corenlp-full-2017-06-09\stanford-corenlp-3.8.0.jar'
# path_to_models_jar = 'path_to/stanford-parser-full-2014-08-27/stanford-parser-3.4.1-models.jar'
path_to_models_jar = r'D:\nlp_lib\stanford-corenlp-full-2017-06-09\stanford-corenlp-3.8.0-models.jar'
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

result = dependency_parser.raw_parse('I shot an elephant in my sleep')
dep = result.next()
list(dep.triples())


""" Dependency Parsing of text.. """

# - exploring the Dependency Grpah part
# Used for parsing text such taht we have (0-9) as a token..
#  [(re.compile('[\([a-z0-9]+\)]'), ' \1 '), (re.compile(r'[\]\[\(\)\{\}\<\>]'), r' \g<0> '), ]
import nltk
tokenizer = nltk.tokenize.TreebankWordTokenizer()
tokenizer.PARENS_BRACKETS = [(re.compile('[\([a-z0-9]+\)]'), ' \1 '), (re.compile(r'[\]\[\(\)\{\}\<\>]'), r' \g<0> ')]
tokenizer.PARENS_BRACKETS = [(re.compile('[\([a-z0-9]\)]'), ' \1 ')]

text = 'EGR1 (henki seenaap) (11) (a mouse embryonic (fibroblasts are the most important cells eva)'
tokenizer.tokenize(text)

# text = 'EGR1(-/-) mouse embryonic fibroblasts'
print(text)
for regexp, substitution in tokenizer.STARTING_QUOTES:
    text = regexp.sub(substitution, text)
for regexp, substitution in tokenizer.PUNCTUATION:
    text = regexp.sub(substitution, text)
# (C35612:) Added this in order to deal with multiple conditions around BRACKETS
for regexp, substitution in tokenizer.PARENS_BRACKETS:
    text = regexp.sub(substitution, text)  # This should be the new one
print(text)

# Now next things we want to do is... analyse each 'part' of the text.
# See if we can get references from it... directives... or maybe label each thing..
# Check this out
import nltk
import re
A = nltk.dependencygraph.DependencyGraph()

test = [x for x in nltk.tokenize.sent_tokenize(all_content[0]) if re.findall('directive', x, re.IGNORECASE)][1]
A = nltk.dependencygraph.DependencyGraph()


import unicodedata
x = 'testing something \xa0\xa0'
y = unicodedata.normalize('NFKD', x)


