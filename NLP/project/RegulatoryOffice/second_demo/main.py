# -*- coding: utf-8 -*-

"""
Second attempt to upgrade the App..

"""

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.label import Label
from kivy.properties import BooleanProperty
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
import project.RegulatoryOffice.fileexplorer as reg_fe
import glob
import os

from embedding.docembedding import TFIDF
from numpy import concatenate, argsort

from kivy.lang import Builder
from kivy.uix.recycleview import RecycleView
from kivy.properties import StringProperty, NumericProperty
from kivy.uix.textinput import TextInput
from random import sample, randint
from string import ascii_lowercase
import project.RegulatoryOffice.ridfunctions as reg_rid
import textprocessing.processtext as text_proc
import importlib
import re

# <editor-fold desc='Defining directories and loading RID data'>
dir_data = r'D:\temp'
dir_reg_doc = dir_data + r'\reg_doc'
src_path = dir_reg_doc + r'\html'
dest_path = dir_reg_doc + r'\html_indexed'
# </editor-fold>


class SelectableRecycleBoxLayout(FocusBehavior, LayoutSelectionBehavior,
                                 RecycleBoxLayout):
    """ Adds selection and focus behaviour to the view."""


class SelectableLabel(RecycleDataViewBehavior, Label):
    """Add selection support to the Label """
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    def refresh_view_attrs(self, rv, index, data):
        """Catch and handle the view changes """
        self.index = index
        return super(SelectableLabel, self).refresh_view_attrs(
            rv, index, data)

    def on_touch_down(self, touch):
        """ Add selection on touch down """
        if super(SelectableLabel, self).on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        """Respond to the selection of items in the view. """
        self.selected = is_selected
        if is_selected:
            print("selection changed to {0}".format(rv.data[index]))
        else:
            print("selection removed for {0}".format(rv.data[index]))


class Test(BoxLayout):
    """ Main Kivy class for creating the initial BoxLayout """

    files_to_index = glob.glob(os.path.join(src_path, '*.html'))
    list_file_name_ext = [os.path.basename(x) for x in files_to_index]
    html_ext = '.html'
    n = len('.html')
    pdf_ext = '.pdf'

    list_file_name = [x[:-len('.html')] if x.endswith('.html') else x[:-len(pdf_ext)]
                      for x in list_file_name_ext]

    list_file_dict = [{'text': x} for x in list_file_name]

    A_JSON = reg_fe.KnowledgeGraph(dir_reg_doc)

    def __init__(self, **kwargs):
        super(Test, self).__init__(**kwargs)

        # Set the list of files we can explore...
        # self.list_file_name  #
        # self.ids.rv.data = self.list_file_dict  # [{'text': str(x)} for x in range(100)]
        self.ids.rv.data = self.list_file_dict  # [{'text': str(x)} for x in range(100)]

    @staticmethod
    def _score_on_tfidf(input_list, k):
        """
        Score list of strings based on tfidf..

        :param input_list: list of sentences
        :param k: top k amount of words are being taken
        :return:
        """

        A = TFIDF(input_list)
        tf_idf_ngram = A.TF_count*A.IDF
        res_conc = concatenate(tf_idf_ngram)
        res_id = concatenate([list(range(A.word_N))] * A.idf_N)
        res_best_id = argsort(-res_conc)
        derp = res_id[res_best_id[:k]]  # This shows me the id for the vocab
        return [A.vocab_dict_num[x] for x in derp]

    @staticmethod
    def _check_content(input_file, ckey):
        """

        :param input_file: input dict with keys
        :param ckey: key to be checked
        :return:
        """
        temp_text = ['No {0} found'.format(ckey.lower())]
        if ckey in input_file.keys():
            temp_text = input_file[ckey]  # Get the keywords
        return temp_text

    def suggest_dir(self):
        """
        Here we make a suggestion for a directoy based on the search criteria
        :param input:
        :return:
        """
        pass

    def change_dir(self):
        # self.filechooser.path = self.text_search.text
        input_query = self.text_search.text
        res_query = self.A_JSON.search(input_query)
        self.ids.rv.data = [{'text': x} for x in res_query]

    def load(self, x):
        """
        Loading data one file at a time.

        :param x:
        :return:
        """
        from bs4 import BeautifulSoup as bS

        y = self.ids.rv.data[x[0]]  # Get the dict data
        dir_file = self.A_JSON.db_json[y['text']]  # Extract the name/celex id
        i_file = dir_file['Raw location']  # Get the location of the file

        bs_obj_file = []
        if i_file.endswith('.html'):
            bs_obj_file = bS(open(i_file, 'rb'), 'lxml')  # open it with bS..
        else:
            print('Here we could analyze the pdf documents...')
            pass

        # Loading the content of the boject.
        self.text_input.text = bs_obj_file.text  # display the text...
        self.text_summary.text = self.return_summary(dir_file)
        self.text_ner.text = self.return_most_common_ner(dir_file)
        self.text_keyword.text = self.return_most_common_keywords(dir_file)
        self.text_trigram.text = self.return_most_common_trigram(dir_file)
        self.text_func.text = self.return_impact(dir_file)

    def load_multiple(self, x, n_max=5):
        # :param n_max: is here to reduce the time to load a document
        # Perform some analyses here..
        from bs4 import BeautifulSoup as bS

        self.derp_multiple_files = [x['text'] for x in self.rv.data]
        # print(self.derp_multiple_files)
        # print(self.A_JSON.db_json[self.derp_multiple_files[0]])

        # self.list_file_name = [self.A_JSON.db_json[x] for x in y]  # Get the dict data
        # print(y)
        # print(self.multi_ngram(5))

        # Loading the content of the boject.
        # self.text_input.text = bs_obj_file.text  # display the text...
        # self.text_summary.text = self.return_summary(dir_file)
        self.text_ner.text = ' '.join(self.multi_ner(5))
        self.text_keyword.text = ' '.join(self.multi_keyword(5))
        self.text_trigram.text = ' '.join(self.multi_trigram(5))
        # self.text_func.text = # self.return_impact(dir_file)

        # list_file = [x['Raw location'] for x in self.list_file_name]  # Get the location of the file
        # print(self.list_file_name)
        # print(list_file)
        # bs_obj_file = []
        # for i_file in list_file[:n_max]:
        #     if i_file.endswith('.html'):
        #         temp = bS(open(i_file, 'rb'), 'lxml')  # open it with bS..
        #     else:
        #         print('Here we could analyze the pdf documents...')
        #         pass
        #     bs_obj_file.append(temp)


        # ALso load information about the file itself...

    def return_summary(self, input_dict):
        """
        Using the AnalyseDoc object to return a summary
        :return:
        """

        # Markup..
        return '\n\n'.join(self._check_content(input_dict, 'Summary'))

    def return_most_common_keywords(self, input_dict):
        """
        Using gensim summarization with TextRank to get some Keywords..
        :return:
        """
        # print('common keywords')
        result_keywords = self._check_content(input_dict, 'Keywords')

        # Markup..
        # s_print = '{:<15} {:>15}\n'.format('keyword', 'score')
        s_print = '{:<15}\n'.format('keyword')
        for x in result_keywords:
            s_print += '\n{:<15}'.format(x)  #, '{:.3f}'.format(i[1]))
            # s_print += '\n{:<15} {:>15}'.format(i[0])  #, '{:.3f}'.format(i[1]))
        return s_print

    def return_most_common_trigram(self, dir_file):
        """

        :return:
        """
        # print('common ngrams')
        # s_print = '{0:{1}}{2}\n'.format('Tri-gram',self.max_len - len('Triplet'), 'Count')
        # s_print = 'Trigram: {0}'.format(x)
        result_trigam = self._check_content(dir_file, 'Trigram')

        # Markup..
        s_print = ''
        for x in result_trigam:
            triplet = ''.join(x)
            # count = x[1]
            # s_print += '\n{0:{1}}{2}'.format(triplet, self.max_len - len(triplet), str(x[1]))
            s_print += '\n{0}'.format(triplet)
            # '{0:{2}}{1:{3}}{4}'.format('test','a',11,5,'henk')
            # Fix this in the Train
            # '{0:{2}}{1:{3}}{4}'.format('test','a',11,5,'henk')
            # '{0:{2}}{1:}'.format('test','a',11)
            # s_print += '\n{:<15} {:>15}'.format(' '.join(x[0]),str(x[1]))
        return s_print

    def return_most_common_ner(self, dir_file):
        """
        :return:
        """
        # print('ner')
        result_ner = self._check_content(dir_file, 'NER')
        # s_print = '{:<20}   {:>20}  {:>20}\n'.format('Entity', 'Category', 'Count')

        # Markup...
        s_print = '{:<20} \n'.format('Entity')
        for x in result_ner:
            # s_print += "\n Entity:\t\"" + '\"\tCategory:\t'.join(x[0]) + '\tCount:\t' + str(x[1])
            # s_print += '\n{:<20} {:>20} {:>20}'.format(x[0][0], x[0][1], x[1])
            s_print += '\n{:<20}'.format(x)
        return s_print

    def return_impact(self, dir_file):
        # print('impact')
        # ugly coding
        derp1 = self._check_content(dir_file, 'Impact on Functions (2nd LOD)')
        if isinstance(derp1, str):
            derp1 = [derp1]
        derp2 = self._check_content(dir_file, 'Impact on Business (1st LOD)')
        if isinstance(derp2, str):
            derp1 = [derp2]
        id_func = [os.path.basename(x) for x in derp1]
        id_biz = [os.path.basename(x) for x in derp2]

        s_print = 'Function: {0} \nBusiness: {1}'.format(', '.join(id_func), ', '.join(id_biz))
        return s_print

    def multi_trigram(self, k):
        """
        Results are based on the predefined json dump...

        :param n:
        :return:

        """
        # Here we concat all the ngrams...
        ngram_string_doc = []
        for x in self.derp_multiple_files:
            input_file = self.A_JSON.db_json[x]
            temp = self._check_content(input_file, 'Trigram')[:k]
            # temp = self.A_JSON.db_json[x]['Trigram'][:k]
            ngram_string = ' '.join([x.replace(' ', '_') for x in temp])
            ngram_string_doc.append(ngram_string)

        res = self._score_on_tfidf(ngram_string_doc, k)
        # res = [x.split('_') for x in res]
        res = [x.replace('_', ' ') for x in res]

        return res  # This shows me the real words..

    def multi_ner(self, k):
        """

        :param k:
        :return:

        """
        # Here we concat all the ngrams...
        ner_string_doc = []
        for x in self.derp_multiple_files:
            input_file = self.A_JSON.db_json[x]
            temp = self._check_content(input_file, 'NER')[:k]
            # temp = self.A_JSON.db_json[x]['NER'][:k]
            ner_string = ' '.join([x.replace(' ', '_') for x in temp])
            ner_string_doc.append(ner_string)

        res = self._score_on_tfidf(ner_string_doc, k)
        # res = [x.split('_') for x in res]

        return res   # This shows me the real words..

    def multi_keyword(self, k):
        """
        Return the ensemble of keywords from multiple documents

        :param k:
        :return:

        """
        # Here we concat all the ngrams...
        keyword_string_doc = []
        for x in self.derp_multiple_files:
            input_file = self.A_JSON.db_json[x]
            temp = self._check_content(input_file, 'Keywords')[:k]
            # temp = self.A_JSON.db_json[x]['Keywords'][:k]
            keyword_string = ' '.join([x for x in temp])
            keyword_string_doc.append(keyword_string)

        res = self._score_on_tfidf(keyword_string_doc, k)
        res = [x.replace('_', ' ') for x in res]

        return res  # This shows me the real words..

class TestApp(App):
    def build(self):
        return Test()


if __name__ == '__main__':
    TestApp().run()