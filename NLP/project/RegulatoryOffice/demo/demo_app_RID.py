# encoding: utf-8

"""

Creating an app with search funcitonalitiy

"""

# <editor-fold desc='Loading all the libraries'>
import xlrd
import os
import re
import gensim

# Import stuff to do some text processing
from bs4 import BeautifulSoup
from textprocessing.processtext import *
import project.RegulatoryOffice.fileexplorer as file_expl

# Import all kind of Kivy stuff...
from kivy.config import Config
Config.set('graphics', 'fullscreen', '0')
from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput

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
col_dict_rev = dict([(v,k) for k,v in col_dict.items()])


""" 
Calculator stuff
"""

class TabTextInput(TextInput):

    def __init__(self, *args, **kwargs):
        self.next = kwargs.pop('next', None)
        super(TabTextInput, self).__init__(*args, **kwargs)

    def set_next(self, next):
        self.next = next

    def _keyboard_on_key_down(self, window, keycode, text, modifiers):
        key, key_str = keycode
        if key in (9, 13):  # and self.next is not None:
            # self.next.focus = True
            # self.next.select_all()
            self.filechooser.path = self.text_search.text
        else:
            super(TabTextInput, self)._keyboard_on_key_down(window, keycode, text, modifiers)


class Root(FloatLayout):
    """
    Derp
    """
    text_loaded = False
    
    stored_summary = ''
    stored_keywords = ''
    stored_ner = ''
    stored_ngram = ''
    max_len = 70
    loaded_file_name = ''

    def suggest_dir(self, input):
        """
        Here we make a suggestion for a directoy based on the search criteria
        :param input:
        :return:
        """
        # self.default_path =


    def change_dir(self):
        self.filechooser.path = self.text_search.text

    def load_multiple(self, path, list_filename):
        """
        Load multiple files at once...
        :param path:
        :param filename:
        :return:
        """
        from bs4 import BeautifulSoup as bS
        list_loaded_file = []
        n = len('.html')
        os.chdir(list_loaded_file)
        for x in list_filename:
            if x.endswith('.html'):
                list_loaded_file = [(bS(open(x, 'rb'), 'lxml'), os.path.basename(x)[:-n]) for x in list_filename]
            else:
                print('Here we could analyze the pdf documents...')
                pass
        return list_loaded_file

    def load(self, path, filename):
        """
        Loading files...

        :param path:
        :param filename:
        :return:
        """
        from bs4 import BeautifulSoup as bS
        x = os.path.join(path, filename[0])
        text_BS = bS(open(x, encoding='utf-8'), 'lxml')

        self.loaded_file_name = filename[0]

        # Prep the obtained text
        # text_extract_obj = file_expl.ExtractRegDoc(text_BS)
        # text_content = [re.sub('\n+','\n',x.text) for x in text_BS.findAll(class_=True) if x['class'][0] == 'normal']
        # text_content = [x for x in text_content if len(x) > 4]

        # self.text_prep = CleanText(text_content, language='english')
        # self.text_lem_stop = self.text_prep.remove_stopwords_text().lemmatize_text()

        # Object to analyse the prepped text
        # self.text_anl = AnalyseDoc(self.text_prep.text)
        # self.text_anl_lem_stop = AnalyseDoc(self.text_lem_stop.text)

        self.text_input.text = text_BS.text
        # file_expl.ExtractRegDoc(text_BS)

    def return_multiple_summary(self):
        """
        Here we build the functionality to return the analysis of a chosen set of files

        :return:
        """
        self.text_summary.text = '\n\n'.join(self.text_anl.get_summary())

    def return_summary(self):
        """
        Using the AnalyseDoc object to return a summary
        :return:
        """
        self.text_summary.text = '\n\n'.join(self.text_anl.get_summary())

    def return_most_common_keywords(self):
        """
        Using gensim summarization with TextRank to get some Keywords..
        :return:
        """
        result_keywords = self.text_anl_lem_stop.most_common_keywords(10)

        s_print = '{:<15} {:>15}\n'.format('keyword','score')
        for i in result_keywords:
            s_print += '\n{:<15} {:>15}'.format(i[0],'{:.3f}'.format(i[1]))
        self.text_keyword.text = s_print

    def return_most_common_ngram(self):
        """

        :return:
        """
        s_print = '{0:{1}}{2}\n'.format('Tri-gram',self.max_len - len('Triplet'), 'Count')

        for x in self.text_anl_lem_stop.most_common_ngrams(5, 3):
            triplet = ' '.join(x[0])
            count = x[1]
            s_print += '\n{0:{1}}{2}'.format(triplet,self.max_len - len(triplet), str(x[1]))
            # '{0:{2}}{1:{3}}{4}'.format('test','a',11,5,'henk')
            # Fix this in the Train
            # '{0:{2}}{1:{3}}{4}'.format('test','a',11,5,'henk')
            # '{0:{2}}{1:}'.format('test','a',11)
            # s_print += '\n{:<15} {:>15}'.format(' '.join(x[0]),str(x[1]))
        self.text_trigram.text = s_print

    def return_most_common_ner(self):
        """
        :return:
        """
        A_most_ner = self.text_anl_lem_stop.most_common_ner(10)
        s_print = '{:<20}   {:>20}  {:>20}\n'.format('Entity','Category','Count')
        for x in A_most_ner:
            # s_print += "\n Entity:\t\"" + '\"\tCategory:\t'.join(x[0]) + '\tCount:\t' + str(x[1])
            s_print += '\n{:<20} {:>20} {:>20}'.format(x[0][0], x[0][1], x[1])
        self.text_ner.text = s_print


    def return_impact(self, data, _col_dict):
        id_url_col = _col_dict['Hyperlink to Regulatory Document']
        id_biz_col = _col_dict['Impact on Business (1st LOD)']
        id_func_col = _col_dict['Impact on Functions (2nd LOD)']

        id_file = os.path.basename(self.loaded_file_name)[6:16]
        id_list = [(i, x) for i, x in enumerate(data[id_url_col]) if re.findall(id_file,x)]
        id_func = [data[id_biz_col][x] for x, y in id_list]
        id_biz = [data[id_func_col][x] for x, y in id_list]

        id_func = split_something(id_func[0])
        id_biz = split_something(id_biz[0])

        self.text_func.text = 'Function: {0} \nBusiness: {1}'.format(', '.join(id_func), ', '.join(id_biz))


class appRID(App):
    pass

if __name__ == '__main__':
    appRID().run()
