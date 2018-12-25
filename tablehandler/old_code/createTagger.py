# encoding: utf-8

"""

In this file we create three classes that will help to load, prepare and label the transaction data. For some example
usage, please use the code below

"""


import pandas as pd
import os
import re
import numpy as np
import glob

import itertools

from tablehandler.old_code.createTable import EditLabels


class HandleFile:
    """
    Used to load and write files
    """

    def __init__(self, year, path_in, month='', path_out=''):
        """
        Initialized the file list based on the year/month and dir input
        Implements methods for reading and writing a list of files

        :param year: Can be string or int, cannot be a list (yet)
        :param path_in: Should be a valid directory, containing all the years-folder of trx_data
        :param month: Can be string or int or list of those types
        :param path_out: Should be a valid directory
        """

        self.year = year
        self.month = month
        self.path_in = path_in
        self.data = np.array([])  # place holder for the data that we load.

        if path_out:
            self.path_out = path_out
        else:
            self.path_out = path_in

        self.file_list = self.get_file_list()

    def get_file_list(self):
        """
        :return: A sorted list of files
        """
        if self.month:
            if isinstance(self.month, list):
                self.month = '(' + '|'.join([str(x).zfill(2) for x in self.month]) + ')'
            else:
                self.month = str(self.month).zfill(2)

        file_dir = glob.glob(self.path_in + '\\*')
        file_list = [x for x in file_dir if re.findall(str(self.year) + '_' + self.month, x)]
        file_list.sort()
        return file_list

    def load_trx_file(self, header=0):
        """
        :param header: If 0, then the first row will be used as a header. Otherwise None
        :return: A list of pandas-read files
        """
        data_trx = [pd.read_csv(x, header=header, parse_dates=True) for x in self.file_list]
        data_trx = [x.fillna('') for x in data_trx]

        self.data = data_trx

        return data_trx

    def write_trx_file(self, data):
        """
        writes data to path_out
        :param data: stuff to be written
        :return: nothing
        """
        for i, x in enumerate(data):
            target_path = os.path.join(self.path_out, os.path.basename(self.file_list[i]))
            x.to_csv(target_path, index=False, header=True)


class PrepRawFile(HandleFile):
    """
    Here we prepare our transaction file and write it to a different folder

    They are moved from raw_transaction to prep_transaction
    """
    def __init__(self, year, path_in, month='', path_out=''):
        """
        Upon initialization, creates the prepped files from file_list
        :param year: see HandleFile
        :param path_in: see HandleFile
        :param month: see HandleFile
        :param path_out: see HandleFile
        """
        super().__init__(year, path_in, month, path_out)
        self.column_names = ["REK_NR", "CUR", "DATE", "DC_IND", "VALUE", "TGN_REK_NR", "NM_TGN_REK", "DATE_2",
                             "MUT_IND", "V1", "DESCR_1", "DESCR_2"]
        self.data = np.array([])
        self.data_prep = np.array([])

    def prep_data(self, data=None):
        """
        converts some columns to dates, adds some columns for years/months as well
        Also converts some columns to upper case only

        :return: a prepped dataframe
        """
        if data:
            assert self.data, "Run load_data() to fill data"
            data = self.data

        data_prep = data[list(range(len(self.column_names)))]
        data_prep.columns = self.column_names
        # Edit date..
        data_prep['DATE_2'] = pd.to_datetime(data_prep['DATE_2'], format="%Y%m%d")
        data_prep['year'] = data_prep['DATE_2'].map(lambda x: str(x.year))
        data_prep['month'] = data_prep['DATE_2'].map(lambda x: str(x.month).zfill(2))
        data_prep['day'] = data_prep['DATE_2'].map(lambda x: str(x.day).zfill(2))
        data_prep['week'] = data_prep['DATE_2'].map(lambda x: str(x.week))  # Why do I need this to be a str?
        data_prep['yearmonth'] = data_prep['year'] + data_prep['month']

        # Edit descr_nr 1
        data_prep['DESCR_1'] = data_prep['DESCR_1'].str.upper()
        data_prep['NM_TGN_REK'] = data_prep['NM_TGN_REK'].str.upper()

        # Edit description nr 2
        reg_string = re.compile(".*([0-9]{2}:[0-9]{2}).*")
        data_prep['DESCR_2'] = [reg_string.sub("\\1", x) for x in data_prep['DESCR_2']]
        data_prep.loc[~data_prep['DESCR_2'].str.contains(reg_string), 'DESCR_2'] = ""

        # Edit value
        data_prep.loc[data_prep.DC_IND == 'D', 'VALUE'] = -data_prep['VALUE']

        # Cum sum
        data_prep['CUM_VALUE'] = np.cumsum(data_prep['VALUE'])

        self.data_prep = data_prep

        return data_prep


class LabelData(HandleFile):
    """
    Used to label data
    """
    def __init__(self, year, path_in, labels, month='', path_out='', n_max=5):
        """
        Upon initialization, already adds the labels, according to 'labels' input, to the file.

        :param year: see HandleFile
        :param path_in: see HandleFile
        :param labels: a json/dict-file that contains labels. See EditLabels class
        :param month: see HandleFile
        :param path_out: see HandleFile
        :param n_max: max size of ngrams of description string
        """
        super().__init__(year, path_in, month, path_out)
        assert isinstance(labels, EditLabels)
        self.data = self.load_trx_file(header=0)
        self.labels = labels
        self.data_label = [self.label_all_cat(x, n_max) for x in self.data]

    def label_cat(self, data, cat, n_max):
        """

        :param data:
        :param cat:
        :param n_max:
        :return:
        """
        input_col = []

        for x in data['DESCR_1']:
            res = set(self.labels.get_query(cat)).intersection(self._descr2ngram(x, n_max))
            input_col.append(', '.join(list(map(self.labels.label_dict[cat].get, list(res)))))

        data[cat] = input_col
        return data

    def label_all_cat(self, data, n_max):
        """
        Labels the data with all categories
        :return:
        """
        for x_cat in self.labels.get_cat():
            data_label = self.label_cat(data, x_cat, n_max)

        return data_label

    def _descr2ngram(self, x, n_max):
        """
        Used to create all ngram options frmo a string

        :param x: input string
        :param n_max: max amount of ngrams taken
        :return: a list with all possible ngrams up to n_max size
        """
        derp = itertools.chain(*[self._find_ngrams(x.split(), i) for i in range(n_max)])
        return set([' '.join(x) for x in derp])

    @staticmethod
    def _find_ngrams(input_list, n):
        """
        Didnt want to use the nltk.ngram thing

        :param input_list: list of string
        :param n: ngram size
        :return: ngrams
        """
        return zip(*[input_list[i:] for i in range(n)])
