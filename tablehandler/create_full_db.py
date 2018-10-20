# encoding: utf-8

"""
Here we are going to create one file from all the transaction documents...

Maybe easier to analyze...

At one point we need to start querieing a database.. because we cant load everything in memory

Lolwait no, it is not even 1 MB yet.
"""

import os
import numpy as np
import re
from tablehandler import createTagger as cTag

dir_transaction = '/home/charmmaria/data/Transacties'

trx_years = [x for x in os.listdir(dir_transaction) if re.match('^[0-9]+$', x)]

for i_year in trx_years:
    temp_path = os.path.join(dir_transaction, i_year)
    trx_files = os.listdir(temp_path)
    temp_raw_files = cTag.PrepRawFile(i_year, path_in=dir_transaction)
