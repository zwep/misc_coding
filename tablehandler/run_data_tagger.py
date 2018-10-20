# encoding: utf-8

"""

With the code below we can tag data

"""

from tablehandler import createTagger as cTag
from tablehandler import createTable as cT
import os
import re

raw_dir = r'D:\data\Testing_data\Transaction\raw_transactions'
prep_dir = r'D:\data\Testing_data\Transaction\prep_transactions'
label_dir = r'D:\data\Testing_data\Transaction\label_transactions'
label_dict_dir = r'D:\data\Testing_data\Transaction\\'

year_set = set([re.sub('.*([0-9]{4}).*','\\1',x) for x in os.listdir(raw_dir)])
labeling_class = cT.EditLabels('label_dict.txt', path=label_dict_dir)

# Upon initialization, the loaded data is immediately preprocessed
for i_year in year_set:
    # Load, prep and write the files
    temp_raw_files = cTag.PrepRawFile(i_year, path_in=raw_dir, path_out=prep_dir)
    temp_raw_files.write_trx_file(temp_raw_files.data_prep)

    test = cTag.HandleFile(i_year, path_in=prep_dir)
    test.load_trx_file()[-12]
    # Load, label and write the files
    temp_label_files = cTag.LabelData(i_year, path_in=prep_dir, labels=labeling_class, path_out=label_dir)
    temp_label_files.write_trx_file(temp_label_files.data_label)
