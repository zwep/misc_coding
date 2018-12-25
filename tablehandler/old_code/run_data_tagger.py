# encoding: utf-8

"""

With the code below we can tag data

"""

import os
import re
import numpy as np
import pandas as pd


def prep_data(data, data_inc):
    """
    converts some columns to dates, adds some columns for years/months as well
    Also converts some columns to upper case only

    :return: a prepped dataframe
    """
    # Edit date..
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


column_names_old = ["IBAN/BBAN","Munt", "Datum","DC_IND", "Bedrag", "Tegenrekening IBAN/BBAN", "Naam tegenpartij",
                    "Rentedatum", "MUT_IND", "V1", "Omschrijving-1", "Omschrijving-2"]
re_comp = re.compile("[0-9]{4}")
year_set = sorted(set([re_comp.findall(x)[0] for x in os.listdir(raw_dir) if re_comp.match(x)]))

for i_year in year_set:
    print(i_year)

    if int(i_year) > 2016:
        ind_header = 0
    else:
        ind_header = None

    year_dir = os.path.join(raw_dir, i_year)
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


        temp = prep_data(A, ind_header)
        #print(prep)

x1 = pd.read_csv(file_path)
x1 = pd.DataFrame(x1)

# [os.path.join(x[0], x[2]) for x in os.walk(raw_dir) if x[2]]
# # labeling_class = cT.EditLabels('label_dict.txt', path=label_dict_dir)



# Upon initialization, the loaded data is immediately preprocessed
for i_year in year_set:
    print('Loading year ', i_year)
    # Load, prep and write the files
    file_list = []
    data_trx = [pd.read_csv(x, header=0, parse_dates=True) for x in file_list]
    data_trx = [x.fillna('') for x in data_trx]

    temp_raw_files = cTag.PrepRawFile(i_year, path_in=raw_dir)
    temp_raw_files.load_trx_file()
    temp_raw_files.data
    temp_raw_files.write_trx_file(temp_raw_files.data_prep)

    test = cTag.HandleFile(i_year, path_in=prep_dir)
    test.load_trx_file()[-12]
    # Load, label and write the files
    temp_label_files = cTag.LabelData(i_year, path_in=prep_dir, labels=labeling_class, path_out=label_dir)
    temp_label_files.write_trx_file(temp_label_files.data_label)
