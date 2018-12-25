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
import pandas as pd


def prep_data(data, i_year, data_inc=None):
    """
    converts some columns to dates, adds some columns for years/months as well
    Also converts some columns to upper case only

    :return: a prepped dataframe
    """
    # Edit date..
    if int(i_year) > 2016:
        data.loc[:, 'Rentedatum'] = pd.to_datetime(data['Rentedatum'], format="%Y-%m-%d")
    else:
        data.loc[:, 'Rentedatum'] = pd.to_datetime(data['Rentedatum'], format="%Y%m%d")
    data['year'] = data['Rentedatum'].map(lambda x: str(x.year))
    data['month'] = data['Rentedatum'].map(lambda x: str(x.month).zfill(2))
    data['day'] = data['Rentedatum'].map(lambda x: str(x.day).zfill(2))
    data['week'] = data['Rentedatum'].map(lambda x: str(x.week))  # Why do I need this to be a str?
    data['yearmonth'] = data['year'] + data['month']

    # Edit descr_nr 1
    data.loc[:, 'Omschrijving-1'] = data['Omschrijving-1'].str.upper()
    data.loc[:, 'Naam tegenpartij'] = data['Naam tegenpartij'].str.upper()

    # Edit description nr 2
    reg_string = re.compile(".*([0-9]{2}:[0-9]{2}).*")
    data['Omschrijving-2'] = data['Omschrijving-2'].fillna('')
    data.loc[:, 'Omschrijving-2'] = [reg_string.sub("\\1", x) for x in data['Omschrijving-2']]
    data.loc[~data['Omschrijving-2'].str.contains(reg_string), 'Omschrijving-2'] = ""

    if data_inc is None:
        A.loc[A['DC_IND'] == 'D', 'Bedrag'] = -A['Bedrag']
    # Cum sum
    data['CUM_VALUE'] = np.cumsum(data['Bedrag'])

    return data


def date_check(input_path):
    z = pd.read_csv(input_path)
    w = prep_data(pd.DataFrame(z))
    return (min(w['DATE']), max(w['DATE']))


dir_transaction = '/home/charmmaria/data/Transacties'
trx_years = [x for x in os.listdir(dir_transaction) if re.match('^[0-9]+$', x)]
# Written in the way the new ones are defined...
column_names_old = ["IBAN/BBAN","Munt", "Datum","DC_IND", "Bedrag", "Tegenrekening IBAN/BBAN", "Naam tegenpartij",
                    "Rentedatum", "MUT_IND", "V1", "Omschrijving-1", "Omschrijving-2"]


for i_year in trx_years:
    temp_path = os.path.join(dir_transaction, i_year)
    trx_files = os.listdir(temp_path)
    i_file = os.path.join(temp_path, trx_files[0])
    temp = pd.read_csv(i_file, encoding='latin')
    print('\n year: ', i_year)
    print('Columns: ', temp.columns)


total_db = pd.DataFrame()

for i_year in sorted(trx_years):
    print(i_year)

    if int(i_year) > 2016:
        ind_header = 0
    else:
        ind_header = None

    year_dir = os.path.join(dir_transaction, i_year)
    year_files = [x for x in sorted(os.listdir(year_dir)) if x.endswith('.txt') or x.endswith('.csv')]

    for i_file in year_files:
        print(i_file)
        file_path = os.path.join(year_dir, i_file)
        A = pd.read_csv(file_path, encoding='latin', header=ind_header)
        A = pd.DataFrame(A)
        print(A.head())
        if int(i_year) < 2017:
            A = A[list(range(len(column_names_old)))]
            A.columns = column_names_old

        temp = prep_data(A, i_year, ind_header)

        total_db = total_db.append(temp)


# https://stackoverflow.com/questions/30631325/writing-to-mysql-database-with-pandas-using-sqlalchemy-to-sql/30653988#30653988
# https://pandas.pydata.org/pandas-docs/stable/io.html#engine-connection-examples
total_db.to_sql()