# encoding: utf-8


"""

Here we are going to fool around with some WhatsApp data to see if we can get some info out of it
"""

import os
import glob
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dir_data = r'\\solon.prd\files\P\Global\Users\C35612\Userdata\Documents\1. Data Innovation ' \
           r'Analytics\Projects\AI-LAB\NLP\Business Case Description\CARE'

name_file = r'\20171001070105-whatsapp-31613660024.csv'

"""

Define helper functions for this script

"""


def x0001_wa(x):
    """ Date  duration for .agg(). Cant be combined with x0002 because of different type output (weird) """
    d = dict()
    d['time_duration'] = x['date'].max() - x['date'].min()
    return pd.Series(d, index=['time_duration'])


def x0002_wa(x):
    """ Aggregation function """
    d = dict()
    d['n_in_ind'] = sum(x['ind'] == 'in')
    d['n_out_ind'] = sum(x['ind'] == 'out')
    d['n_count'] = int(len(x))  # [len(x)] * len(x['date'])
    d['n_words'] = len(nltk.tokenize.word_tokenize(' '.join(x['message'])))
    # d['n_words'] = ' '.join(x['message'])
    return pd.Series(d, index=['n_in_ind', 'n_out_ind', 'n_count', 'n_words'])


"""

Function for preprocessing whatsapp data

"""


def wa_prep(data):
    """

    :param data:
    :return:
    """

    data['message'] = data.message.fillna('')
    # Make sure that we know at all times to whom the caller is speaking
    # This is necessary for this split of conversations
    data['email'] = data.groupby('telnr_in').email.fillna(method='backfill')

    # Convert date to dates
    data['date'] = pd.to_datetime(data.date)

    data['conv_id'] = data.groupby(['telnr_in', 'email']).grouper.group_info[0]

    # Try to get 'sessions' of conversations
    # By subtracting max and min date
    time_split = data.groupby(['telnr_in', 'email']).apply(x0001_wa)
    time_sec = [x.total_seconds() for x in time_split.time_duration]  # Convert to total amount of seconds
    time_split['sec'] = time_sec
    data = pd.merge(data, time_split, left_on=['telnr_in', 'email'], right_index=True)

    # data in-out specifier per conversation
    d_conv = {'in': 1, 'out': 0}
    z = np.insert(np.abs(np.diff(data.ind.apply(lambda x: d_conv.get(x, x)).values)), 0, 0)
    data['ind_flow'] = np.cumsum(z)

    # This will help us to locate indiviual messages over multiple lines
    text_added = data.groupby(['telnr_in', 'email', 'ind_flow']).message.agg(lambda col: '. '.join(col))
    text_added.names = 'message_concat'
    data = pd.merge(data, text_added.to_frame(), left_on=['telnr_in', 'email', 'ind_flow'], right_index=True)

    # Here we have a subset of the initial table where all conversations are (per input) concatten
    # Such that we have a sort-of-question-answer relation
    data_subset = data[['conv_id', 'sec', 'ind_flow', 'message_y']].drop_duplicates()

    return data_subset


""" 
Loading data
"""

# We need to read it in using fixed columns.. otherwise the text is messed up
whatsapp_col = ['date', 'telnr_in', 'ind', 'status', 'type', 'message', 'email', 'unk']
list_file_name = glob.glob(dir_data + '\*.csv')

list_data = []
for i_file in list_file_name:
    print(os.path.basename(i_file))
    A = pd.read_csv(i_file, names=whatsapp_col, index_col=False)
    list_data.append(A)

# Check for which months we have data..
for x in [(min(x.date), max(x.date)) for x in list_data]:
    print(x)

whatsapp_data = pd.concat(list_data).reset_index()

""" 
Preprocessing
"""

clean_data = wa_prep(whatsapp_data)


"""
Conversation splitting on time spend
"""

# Some proposed split after a first analysis with histograms..
# Some conversations are long (couple of days)
# Some are too short (either missing the conversation, or a test by the employees to check the system)\
some_perc = [0, 50, 75, 85, 100]
sec_split = [np.percentile(clean_data.sec, x) for x in some_perc]  # Ja dit is dus vrij random eigenlijk..

for i in range(len(sec_split)):
    m, s = divmod(sec_split[i], 60)
    h, m = divmod(m, 60)
    print('Percentile value: ', some_perc[i], ' Represents (H:M:S): %d:%02d:%02d' % (h, m, s))

# Based on these percentiles.. we are going to split it into the following categories
# - Quick conversations
# - Medium1
# - Medium2
# - Longer
# Could think of  better names of course. But screw that, aI Aint No Marketeerr

clean_data_sub = []
for i in range(len(sec_split)-1):
    cond_1 = (clean_data.sec < sec_split[i+1])
    cond_2 = (clean_data.sec > sec_split[i])
    B = clean_data.loc[cond_1 * cond_2]
    clean_data_sub.append(B)


"""

Conversation counts... statistics and stuff

"""

# Apply some operations here on the subsetted data..

# Counting parts of the conversation
# - how much messages were sent in/out
# - total length (irrelavant actually)
# - amount of words in the message (can be split to avg. in and avg. out)
# - Response time
# - Interarrival time distribution over the conversation -> How 'fast' is the conversation going
