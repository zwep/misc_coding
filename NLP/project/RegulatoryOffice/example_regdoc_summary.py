# encoding: utf-8

"""
Here we show how to summarize the regulatory documents

"""

from project.RegulatoryOffice.ridfunctions import get_RID
from project.RegulatoryOffice.ridfunctions import get_html_obj
import project.RegulatoryOffice.fileexplorer as file_exp

# Self created summarizer
from summarization.frequencysummarizer import FrequencySummarizer
from summarizer import summarize  # The 'github' summarizer
# Innerworkings: https://github.com/michigan-com/summarizer/blob/master/summarizer/summarizer.py
from gensim.summarization.summarizer import summarize as TextRank
from gensim.summarization.keywords import keywords as TextRankKeywords

import importlib

# <editor-fold desc='Defining locations and loading data'>
dir_data = r'D:\temp'
dir_reg_doc = dir_data + r'\reg_doc'
dir_reg_html = dir_reg_doc + r'\html'
name_data = 'Export_RID_16012018.xlsx'

rid_data = get_RID(name_data, dir_reg_doc)  # Gives back a dictionaru
n_obj = 5
html_obj = get_html_obj(dir_reg_html, n_obj)  # Gives back a tuple of (content, name)
# </editor-fold>

importlib.reload(file_exp)
# Here we apply the summarizer on the first block of text
B = file_exp.ExtractRegDoc(*html_obj[0], rid_data=rid_data)

first_block_text = [x.text for x in B.get_first_block() if len(x.text) > 3]
last_block_text = [x.text for x in B.get_last_block() if len(x.text) > 3]

fs = FrequencySummarizer()
fs.summarize(first_block_text, 3)
fs.summarize(last_block_text, 3)

# Trying the github summarizer
# Location: print(summarizer.__file__)

summarize('Directive', ' '.join(first_block_text), 3)

# Trying TextRank
# This one is available in gensim
TextRank(' '.join(first_block_text), 3)
TextRankKeywords(' '.join(first_block_text)).split('\n')