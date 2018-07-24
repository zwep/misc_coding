# encoding: utf-8

"""


"""

# <editor-fold desc='Loading all the libs'>

# Import stuff to do some text processing
import xlrd
from textprocessing.processtext import *

import re

from langdetect import detect

# </editor-fold>

"""
Loading RID data for reference
"""


dir_data = r'D:\temp\reg_doc'
name_data = '\Export_RID_16012018.xlsx'

workbook = xlrd.open_workbook(dir_data + name_data)
sheet = workbook.sheet_by_index(0)

col_names = [x.value for x in sheet.row(0)]
# Based on the printed statements above, choose the following columns
sel_col = [2, 4, 10, 11, 14, 19, 20, 24, 26, 27, 28, 29]

xls_data = [sheet.col_values(i_col) for i_col in sel_col]
col_dict = dict([(i, xls_data[i][0]) for i in range(len(xls_data))])
col_dict_rev = dict([(v, k) for k, v in col_dict.items()])

# return_impact(xls_data, col_dict_rev, '32016R1450')


def return_col_overview(data):
    """ Stupid doc thing """
    for i_col in data:
        print('\n-----------', i_col[0], '-------------\n')

        # This is needed to remove the names from the impacted regions and also to split the multiple regions.. but in an
        #  appropiate way ) without losing information.

        if re.findall('Impact', i_col[0]):
            i_col_temp = []
            for x in i_col:
                derp = []
                for i_item in x.split(','):
                    derp.append(re.sub('\(.*\)', '', i_item).strip())
                if len(derp) == 1:
                    i_col_temp.append(derp[0])
                else:
                    # Here we can also choose to append the tuple...
                    # Or we can just treat this as an extra case and extend it..
                    i_col_temp.extend(derp)

            # Update the column
            i_col = i_col_temp

        z_collect = collections.Counter(i_col)
        z_common = z_collect.most_common(10)

        for i in z_common:
            print(i, '- {:.2f}%'.format(i[1]/sum(z_collect.values())*100))
        try:
            input()
        except KeyboardInterrupt:
            print('User interrupted')
            break


def return_language_dist(data, _col_dict):

    # Checking language distribution of the short description together with the country thingy
    id_col = _col_dict['Short Description']
    lang_desc = [detect(x) if len(x) != 0 else 'unk' for x in data[id_col]]
    lang_combi_desc = collections.Counter(zip(data[1], lang_desc))
    for i, i_count in lang_combi_desc.most_common(100):
        print(i, 'occurence count: ', i_count)


def return_impact(data, _col_dict, id_file):
        id_url_col = _col_dict['Hyperlink to Regulatory Document']
        id_biz_col = _col_dict['Impact on Business (1st LOD)']
        id_func_col = _col_dict['Impact on Functions (2nd LOD)']

        id_list = [(i, x) for i, x in enumerate(data[id_url_col]) if re.findall(id_file,x)]
        id_func = [data[id_biz_col][x] for x, y in id_list]
        id_biz = [data[id_func_col][x] for x, y in id_list]
        return (id_func, id_biz)


test = [x for i, x in enumerate(xls_data) if i in [1, 5, 6, 8, 9, 10]]
return_col_overview(test)
return_language_dist(xls_data, col_dict_rev)
