# encoding: utf-8

"""

Here we define a set of functions/classes that help to extract information from the RID. The two categories are

- Downloading html info based on RID urls (EURLEX)
- Extracting impact information from RID based on the found htmls (EURLEX)
"""


# <editor-fold desc='Loading lib'>

import collections
import glob
import itertools
from langdetect import detect
import os
import re
import requests
import shutil

# </editor-fold>


# <editor-fold desc='Functions'>
def get_RID(name, path=os.getcwd()):
    """
    Returns the RID data as a dict. Not interested in using Pandas for this...

    only thing is that the xlrd thing reads dates as some integer... could be fixed later if necessary

    :param name:
    :param path: input of the path...
    :return:
    """
    import xlrd
    workbook = xlrd.open_workbook(path + '\\' + name)
    sheet = workbook.sheet_by_index(0)
    rid_data = [sheet.col_values(i_col) for i_col in range(sheet.ncols)]
    rid_data_dict = {i_col[0]: i_col[1:] for i_col in rid_data}

    return rid_data_dict


def return_language_dist(data, _col_dict):
    """
    Checking language distribution of the short description together with the country thingy

    :param data:
    :param _col_dict:
    :return:
    """
    id_col = _col_dict['Short Description']
    lang_desc = [detect(x) if len(x) != 0 else 'unk' for x in data[id_col]]
    lang_combi_desc = collections.Counter(zip(data[1], lang_desc))
    for i, i_count in lang_combi_desc.most_common(100):
        print(i, 'occurence count: ', i_count)


def analyse_RID(rid_data):
    """
    Returns content from the dict that represents the RID data

    :param rid_data: dict-type
    :return:
    """
    for col_name, i_col in rid_data.items():
        print('\n-----------', col_name, '-------------\n')

    # This is needed to remove the names from the impacted regions and also to split the multiple regions.. but in an
    #  appropiate way ) without losing information.

        if re.findall('Impact', col_name):
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


def get_html_url_from_RID(rid_data):
    """
    Returns the Celex ids from the RID database

    :param rid_data: the database from RID... should contain atleast the column 'Hyperlink to Regulatory Document'
    :return: Returns a 3-tuple with index in the RID database (Row-id), celex number, and file name.
    """
    url_list = rid_data['Hyperlink to Regulatory Document']
    url_list_valid = [(i, x) for i, x in enumerate(url_list) if x[0:4] == 'http']  # Make sure that we get a website

# i_url = [x[1] for x in url_list_valid if x[0] == 1260][0]

    html_url_eurlex = []
    for i, i_url in url_list_valid:
        if re.findall('CELEX', i_url):
            index_rid = i
            # url_eurlex_rid = re.search('CELEX:([0-9]{5}[A-Z](.{1}|.{2})[0-9](.{3}|.{4}))', i_url).group(1)
            # url_eurlex_rid = re.search('CELEX:(.*)(\&|$)', i_url).group(1)
            # url_eurlex_rid = re.search('CELEX:([a-zA-Z0-9]*)(\&|$)', i_url).group(1)
            url_eurlex_rid = re.search('CELEX:(.+?)(?=&|$)', i_url).group(1)
            # celex_id = 'CELEX_' + url_eurlex_rid + '.html'
            celex_id = url_eurlex_rid + '.html'  # We are not going to use the celex prefix anymore.
            # print(i, celex_id)
            html_url_eurlex.append((index_rid, url_eurlex_rid, celex_id))

    return html_url_eurlex


def call_eurlex_api(query, username, password):
    """
    This makes a call to the EURLEX database given username and password. Because scraping was not allowed.

    :param query:
    :param username:
    :param password:
    :return:

    TODO you still need to work on this one...
    """

    query_eurlex = 'CELEX_number = 32014L0065'
    query_header = {'username': 'n002033a', 'password': 'JuLt7Fhm8pQ', 'query': query}
    wsdl = 'http://eur-lex.europa.eu/eurlex-ws?wsdl'
    res = requests.get(wsdl, headers=query_header)

    if res.status_code == 200:
        return res
    else:
        print('Something went wrong... Status code:', res.status_code)
        print('Using this query:', query)


def save_html_object(html_obj, path):
    """
    With this we can save the html objects to a specific path

    :param html_obj: a list of tuples.. first one is the content, second item is the name
    :param path:
    :return:
    """
    for i_obj, obj_name in html_obj:
        with open(path + '\\' + obj_name, 'wb') as f:
            f.write(i_obj.text.encode())


def load_html_obj(path, n=None):
    """
    Returns a list of html objects, given the path

    :param path: path to html objects
    :param n: subsets the amount of html objects loaded
    :return: a list of Beautiful Soup objects corresponding to the htmls
    """
    from bs4 import BeautifulSoup
    import glob
    if n is None:
        url_html_list = glob.glob(path + "\*html")
    else:
        url_html_list = glob.glob(path + "\*html")[:n]

    return [(BeautifulSoup(open(x, 'rb'), 'lxml'), os.path.basename(x)[:-5]) for x in url_html_list]

# </editor-fold>


# <editor-fold desc='Classes'>
class ExtractRidData:
    """
    Prep some stuff...

    TODO make a dict with key:value pairs as in eurlex0id: propdict..
    """

    col_imp_fun = 'Impact on Functions (2nd LOD)'
    dir_imp_fun = 'Impact on Functions'

    col_reg_file = 'Link to Existing Regulatory File'
    dir_reg_file = 'Regulatory File'

    col_reg_topic = 'Regulatory Topic'
    dir_reg_topic = 'Regulatory Topic'

    col_reg_type = 'Regulation Type'
    dir_reg_type = 'Regulation Type'

    col_imp_bus = 'Impact on Business (1st LOD)'
    dir_imp_bus = 'Impact on Business'

    col_imp_cty = 'Impact on Countries'
    dir_imp_cty = 'Impact on Countries'

    col_sel = ['Impact on Functions (2nd LOD)', 'Link to Existing Regulatory File', 'Regulatory Topic',
                'Regulation Type', 'Impact on Business (1st LOD)', 'Impact on Countries']

    def __init__(self, dir_file, rid_data):
        """

        :param dir_file: list of files, result of glob.glob(...)
        :param rid_data:
        """
        # Take only the file name
        self.dir_file = dir_file
        self.file_name, self.file_ext = zip(*[(os.path.basename(os.path.splitext(x)[0]), os.path.splitext(x)[1]) for x
                                              in self.dir_file])

        # Extract the html-urls from the rid data
        self.celex_file_combo = get_html_url_from_RID(rid_data)
        # Unzip it to obtain the celex ids and ...
        _, self.celex_id, self.celex_id_ext = zip(*self.celex_file_combo)

        # Subset the rid_data dict based on column/key values
        rid_data_sel_col = {k: v for k, v in rid_data.items() if k in self.col_sel}

        # Shows per file_name entry to which index it is related in RID
        id_subset_list = [[i for i, y in enumerate(self.celex_id) if y in x] for x in self.file_name]

        # Subset the rid_data based on the found CELEX ids (overall)
        # In return we get a list of dicts.. with the same length as the files that we have put in...
        sel_rid_data = self.subset_dict(rid_data_sel_col, id_subset_list)
        # Here we turn it into a dict again with the CELEX ids as keys.
        sel_rid_data = dict(zip(self.file_name, sel_rid_data))

        self.prep_rid_data = self._prep_rid_data(sel_rid_data)

    @staticmethod
    def subset_dict(input_dict, id_list, join_param=', '):
        """
        A method in order to subset the rid data in a more... clear version

        :param input_dict: an input dictionary with list as values (e.g. dict('k1': [1,2,3,4,5], 'k2': [4,5,6,7])
        :param id_list: a list of list with id's that relate to one entity (e.g. [[1,2,3],[4,5],[1,2,5]]
        :param join_param: used to join the multiple row values that are found in the dict
        :return: a list of dict with the values (list subset) concat by the ', '.join() operation
        """
        dict_list = []
        for row_id_list in id_list:
            concat_dict = {k: join_param.join([str(v[row_id]) for row_id in row_id_list]) for k, v in input_dict.items()}
            dict_list.append(concat_dict)
        return dict_list

    def _prep_rid_data(self, rid_data):
        """
        Combination of all the prep functions on the different columns

        :param rid_data:
        :return:
        """

        rid_data = self._prep_impact_business(rid_data)
        rid_data = self._prep_impact_country(rid_data)
        rid_data = self._prep_impact_function(rid_data)
        rid_data = self._prep_link_existing_reg_file(rid_data)
        rid_data = self._prep_regulation_type(rid_data)
        rid_data = self._prep_regulatory_topic(rid_data)
        return rid_data

    def _prep_impact_business(self, input_dict):
        """
        Tested

        :return:
        """
        for k_celex_id, value_dict in input_dict.items():
            i_row = []
            for y in value_dict[self.col_imp_bus].split(','):  # Split multiple topics
                sub_row = []
                for z in re.split('- | -', y):  # Split based on topic and sub-topic
                    k = re.sub('\(.*\)', '', z).strip()
                    if len(k) == 0:
                        k = 'Unkown'
                    sub_row.append(k)
                z = '\\'.join(sub_row)
                if self.dir_imp_bus not in z:
                    z = self.dir_imp_bus + '\\' + z
                i_row.append(z)
            # input_dict[i][self.col_imp_bus] = ', '.join(list(set(i_row)))
            input_dict[k_celex_id][self.col_imp_bus] = list(set(i_row))

        return input_dict

    def _prep_impact_country(self, input_dict):
        """
        Test

        :return:
        """
        for k_celex_id, value_dict in input_dict.items():
            i_row = []
            for y in value_dict[self.col_imp_cty].split(','):  # Split multiple topics
                sub_row = []
                for z in re.split('- | -', y):  # Split based on topic and sub-topic
                    k = re.sub('\(.*\)', '', z).strip()
                    if len(k) == 0:
                        k = 'Unkown'
                    sub_row.append(k)
                z = '\\'.join(sub_row)
                if self.dir_imp_cty not in z:
                    z = self.dir_imp_cty + '\\' + z
                i_row.append(z)
            # input_dict[i][self.col_imp_cty] = ', '.join(list(set(i_row)))
            input_dict[k_celex_id][self.col_imp_cty] = list(set(i_row))

        return input_dict

    def _prep_impact_function(self, input_dict):
        """
        Tested

        :param input_dict:
        :return:
        """
        for k_celex_id, value_dict in input_dict.items():
            i_row = []
            for y in value_dict[self.col_imp_fun].split(','):  # Split multiple topics
                k = re.sub('\(.*\)', '', y).strip()
                if k == '':
                    k = 'Unknown'
                if self.dir_imp_fun not in k:
                    k = self.dir_imp_fun + '\\' + k
                i_row.append(k)
            # input_dict[i][self.col_imp_fun] = ', '.join(list(set(i_row)))
            input_dict[k_celex_id][self.col_imp_fun] = list(set(i_row))

        return input_dict

    def _prep_link_existing_reg_file(self, input_dict):
        """
        Tested

        :param input_dict:
        :return:
        """
        for k_celex_id, value_dict in input_dict.items():
            i_row = []
            for y in value_dict[self.col_reg_file].split(','):  # Split when multiple files are involved
                k = re.sub('/', '_', re.split('- | -', y)[0].strip())
                if k == '' or k == '-':
                    k = 'Unknown'
                if self.dir_reg_file not in k:
                    k = self.dir_reg_file + '\\' + k
                i_row.append(k)
            # input_dict[i][self.col_reg_file] = ', '.join(list(set(i_row)))
            input_dict[k_celex_id][self.col_reg_file] = list(set(i_row))

        return input_dict

    def _prep_regulation_type(self, input_dict):
        """

        :return:
        """

        for k_celex_id, value_dict in input_dict.items():
            i_row = []
            for y in value_dict[self.col_reg_type].split(','):  # Split multiple topics
                k = re.sub('/', '_', y).strip()
                if len(k) == 0:
                    k = 'Unkown'
                if self.dir_reg_type not in k:
                    k = self.dir_reg_type + '\\' + k
                i_row.append(k)
            # input_dict[i][self.col_reg_type] = ', '.join(list(set(i_row)))
            input_dict[k_celex_id][self.col_reg_type] = list(set(i_row))

        return input_dict

    def _prep_regulatory_topic(self, input_dict):
        """
        Tested

        :return:
        """
        for k_celex_id, value_dict in input_dict.items():
            i_row = []
            for y in value_dict[self.col_reg_topic].split(','):  # Split multiple topics
                sub_row = []
                for z in re.split('- | -', y):  # Split based on topic and sub-topic
                    k = z.strip()  # Strip where necessary
                    if len(k) == 0:
                        k = 'Unkown'
                    sub_row.append(k)
                z = '\\'.join(sub_row)
                if self.dir_reg_topic not in z:
                    z = self.dir_reg_topic + '\\' + z
                i_row.append(z)
            # input_dict[i][self.col_reg_topic] = ', '.join(list(set(i_row)))
            input_dict[k_celex_id][self.col_reg_topic] = list(set(i_row))

        return input_dict
# </editor-fold>
