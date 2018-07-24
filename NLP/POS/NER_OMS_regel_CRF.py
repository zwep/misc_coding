from itertools import accumulate
import pycrfsuite
from collections import Counter

import os
import pandas as pd
import re
import numpy as np


location_oms = r'D:\data\text\OMSregel'

os.chdir(location_oms)
A = pd.read_csv('raw_OMS_regel_10000.csv')
# A = pd.read_csv('raw_OMS_regel_10000000.csv')
A = A['oms_regel_1']
# Make sure that we have an extra space here
A = [re.sub('(/[0-9]{2}\.[0-9]{2})', '\\1 ', x) for x in A]

# We have tested that replacing * with ' ' is OK. It separates the CCV and other terminal stuff
# We have also looked at -, this is a bit more dificult.. but not too bad for shop names (different for location names)
almost_clean_data = [re.sub('-', '', re.sub('\*', ' ', x[32:-7])) for x in A]
clean_data = [re.sub('\s+', ' ', re.sub('([0-9]+)', ' \\1 ', x)).strip() for x in almost_clean_data]

shop_labels = {'albert heijn', 'h&m', 'blokker', 'tinq', 'action', 'kruidvat', 'etos', 'bakkerij kerssens', 'praxis',
               'bouwmarkt', 'nijvendal', 'eye wish', 'dekamarkt', 'vomar', '4 yourhair', 'jumbo', 'coop', 'plus',
               'zeeman', 'kwik fit', 'aldi', 'kfc', 'bruna', 'Het Goed', 'lochan supermarkt', 'wolff', 'primera',
               'total', 'ikea', 'happy italy', 'hema', 'flying tiger', 'zara', 'texaco', 'shell', 'lidl', 'sligro',
               'dirk vdbroek', 'mitra', 'simon levelt', 'pathe', 'gall & gall', 'bakkerij vink', 'leonidas', 'ah',
               'sodexo', 'v&d', 'hoogvliet'}

label_data = [[x for x in shop_labels if len(re.findall(x, y.lower()))] for y in clean_data]
label_data = [x[0].split(' ') if len(x) > 0 else '' for x in label_data]

"""
# Check which things you ar emissing
# Works pretty simple. just count all the words that you missed. Track the most occuring ones.
a = [y for i_index, y in enumerate(clean_data) if label_data[i_index] == '']
b = [x.split() for x in a]
x1, x2 = np.unique([x for a in b for x in a], return_counts=True)
A = pd.DataFrame([x1,x2]).transpose()
A.columns = ['x1', 'x2']
A = A.sort_values('x2')
A[-50::]
"""


# here we can either use the tokenizer... or just split by blanks..
# token_data = [nltk.tokenize.word_tokenize(x, language='dutch') for x in clean_data]
token_data = [x.split(' ') for x in clean_data]

# define a separate one with lowercase because the cased one can be interesting for later analyusis
token_data_lower = [[b.lower() for b in x] for x in token_data]

# This makes the target set immediately
ind_label_data = [[str(x in y) for x in token_data_lower[i_index]] for i_index, y in enumerate(label_data)]

# Here we get the index of the 'tokenzied' sentence in OMS regel
# HOwever this only covers singular labels.... as 'h&m'
# fuck this. super annoing with lists and values
# index_label_data = [x.index(y[0]) if y[0] in x else '' for x, y in zip(token_data_lower, label_data)]

# Here we are going to subset data based on found patterns
bool_label_data_slice = [x != '' for x in label_data]

ind_label_data_slice = [x for i, x in enumerate(ind_label_data) if bool_label_data_slice[i]]
token_data_lower_slice = [x for i, x in enumerate(token_data_lower) if bool_label_data_slice[i]]
token_data_slice = [x for i, x in enumerate(token_data) if bool_label_data_slice[i]]
# Try to make these to check whether it generalizes in a proper way... because that is wahat we want
unseen_ind_label_data_slice = [x for i, x in enumerate(ind_label_data) if not bool_label_data_slice[i]]
unseen_token_data_lower_slice = [x for i, x in enumerate(token_data_lower) if not bool_label_data_slice[i]]
unseen_token_data_slice = [x for i, x in enumerate(token_data) if not bool_label_data_slice[i]]

# Continue with setting up the train set
total_lower_set = list(map(list, map(zip, token_data_lower_slice, ind_label_data_slice)))
unseen_total_lower_set = list(map(list, map(zip, unseen_token_data_lower_slice, unseen_ind_label_data_slice)))

total_set = list(map(list, map(zip, token_data_slice, ind_label_data_slice)))
unseen_total_set = list(map(list, map(zip, unseen_token_data_slice, unseen_ind_label_data_slice)))


p_train = 0.7
n = len(total_set)
n_train = int(np.round(n*p_train))
index_full = range(n)
index_train = np.random.choice(index_full, n_train, replace=False)
index_test = list(set.difference(set(index_full), set(index_train)))

# Super gay dit
train_set = list(np.array(total_set)[index_train])
test_set = list(np.array(total_set)[index_test])


# nltk.corpus.conll2002.fileids()
# train_sents = list(nltk.corpus.conll2002.iob_sents('ned.train'))
# test_sents = list(nltk.corpus.conll2002.iob_sents('ned.testb'))


def word2features(sent, wrd_strt, wrd_nd, i):
    # Sent is the sentence split up into a list of tuples
    # Here we take tuple i and by choosing [0] we get the word
    word = sent[i][0]
    word_start = str(wrd_strt[i])
    word_end = str(wrd_nd[i])

    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.firstisupper=%s' % word[0].isupper(),
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'word.startdigit=%s' % word_start,
        'word.enddigit=%s' % word_end,
    ]
    if i > 0:
        word1 = sent[i-1][0]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            ])
    else:
        features.append('BOS')

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            ])
    else:
        features.append('EOS')

    return features


def sent2features(sent):
    # sent = train_set[8]
    word_length = list(map(len, sent2tokens(sent)))
    word_end_index = list(accumulate(word_length))
    word_start_index = [x1 - x2 for x1, x2 in zip(word_end_index, word_length)]
    return [word2features(sent, word_start_index, word_end_index, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label in sent]


def sent2tokens(sent):
    return [token for token, label in sent]

X_train = [sent2features(s) for s in train_set]
y_train = [sent2labels(s) for s in train_set]

# Still need to create a test set
X_test = [sent2features(s) for s in test_set]
y_test = [sent2labels(s) for s in test_set]

trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})


# This train the model and saves it to a file!
trainer.train('oms_regel_train.crfsuite')


# Now we move on to predict.. hence we open the crfsuite object again
tagger = pycrfsuite.Tagger()
tagger.open('oms_regel_train.crfsuite')
example_sent = test_set[5]

#
print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
print("Correct:  ", ' '.join(sent2labels(example_sent)))


# Make better generalizing features I guess
for i in range(100):
    print('-----------------------')
    unseen_example_sent = unseen_total_set[i]
    print("Predicted:", ' '.join(tagger.tag(sent2features(unseen_example_sent))))
    print("Correct:  ", ' '.join(sent2tokens(unseen_example_sent)))

y_pred = [tagger.tag(sent2features(x)) for x in test_set[0:10]]

info = tagger.info()


def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(info.transitions).most_common(15))

print("\nTop unlikely transitions:")
print_transitions(Counter(info.transitions).most_common()[-15:])


def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))

print("Top positive:")
print_state_features(Counter(info.state_features).most_common(20))

print("\nTop negative:")
print_state_features(Counter(info.state_features).most_common()[-20:])





