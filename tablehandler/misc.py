#
import re
import itertools
import collections
import numpy as np
import pandas as pd
import os

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def get_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))

data_dir = r'C:\Users\20184098\Documents\data\transacties'
name_file = 'CSV_A_20190209_161043.csv'

A = pd.read_csv(os.path.join(data_dir, name_file), encoding='latin')

# Some data prep...
A['Bedrag'] = A['Bedrag'].str.replace(',', '.').astype(float)
A['Cum_Bedrag'] = A['Bedrag'].cumsum()
A['Time'] = A['Omschrijving-1'].map(lambda x: re.findall('([0-9]{2}:[0-9]{2})', x))
A['Time'] = A['Time'].map(lambda x: x[0] if len(x) else '0:00')

A['Full_date'] = pd.to_datetime(A['Datum'].map(str) + ' ' + A['Time'].map(str))
A['Datum'] = pd.to_datetime(A['Datum'])
A['Time'] = pd.to_datetime(A['Time'])
A['yearmonth'] = A['Datum'].map(lambda x: str(x.year) + '-' + str(x.month).zfill(2))

# Selection...
A_sel_maand = A[A['Datum'] > pd.datetime.strptime('2019-01-01', '%Y-%M-%d')]

# Text exploration to come with all the name variants..
# This is then used to create certain files for labeling..
descr_values = [str(x).split() for x in A['Naam tegenpartij'].to_list()]
descr_values_2gram = [get_ngrams(x, 2) for x in descr_values]
descr_values_3gram = [get_ngrams(x, 3) for x in descr_values]

w = list(itertools.chain(*descr_values))
collections.Counter(w).most_common()

w = list(itertools.chain(*descr_values_2gram))
collections.Counter(w).most_common()

w = list(itertools.chain(*descr_values_3gram))
collections.Counter(w).most_common()


def get_labels(x):
    pass
def get_statistics(A):
    pass

np.min(A['Datum'])
np.max(A['Datum'])

def get_transaction_count(A):
    z0 = A.groupby(['Full_date', 'Time']).count().reset_index(1).groupby('Time').sum()['Munt']
    z0 = z0[~(z0 == np.max(z0))]
    z0_mean = z0.rolling(10, min_periods=1).mean()
    return z0_mean

z0_mean_overal = get_transaction_count(A)
z0_mean_overal = z0_mean_overal/np.max(z0_mean_overal)
z0_mean_sel = get_transaction_count(A_sel_maand)
z0_mean_sel = z0_mean_sel/len(z0_mean_sel)
# Hier iets mee doen zodat je je maandelijkse uitgaven kan vergelijken met t gemiddelde patroon..
ax = z0_mean_overal.plot(style='b-')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H %M'))
ax = z0_mean_sel.plot(style='r*')



# Read other files and check content..?
old_data_dir = r'C:\Users\20184098\Documents\data\transacties'
