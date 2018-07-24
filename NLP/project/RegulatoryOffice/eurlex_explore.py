# encoding: utf-8

"""
In this file we are trying to parse an HTML from some legal document...

such that we can loop over them alinea by
alinea... or article by article. In the hopes that the html document will contain better organized structure of the
document itself.

The final goal is to make sure that we can make some kind of reference graph, where we know which piece of text
refers that what.

TODO should be merged with eurlex_analysis...
"""

import requests
import importlib

import project.RegulatoryOffice.fileexplorer as reg_fe
import project.RegulatoryOffice.ridfunctions as proc_rid
import textprocessing.processtext as proc_text

from helper.miscfunction import color_back_red, n_diff_cluster

dir_data = r'D:\temp'
dir_reg_doc = dir_data + r'\reg_doc'
dir_reg_html = dir_reg_doc + r'\html'
name_data = 'Export_RID_16012018.xlsx'

rid_data = proc_rid.get_RID(name_data, dir_reg_doc)  # Gives back a dictionary
html_obj = proc_rid.load_html_obj(dir_reg_html, 5)  # Gives back a tuple of (content, name)


A = reg_fe.ExtractRegDoc(html_obj[0], rid_data=rid_data)
ref_html_dir, ref_celex_dir = zip(*A.transform_to_celex(A.get_reference_directives(), 'Directive'))
ref_html_reg, ref_celex_reg = zip(*A.transform_to_celex(A.get_reference_regulation(), 'Regulation'))


extract_html_obj = []
for i_obj in html_obj:
    extract_html_obj.append(reg_fe.ExtractRegDoc(*i_obj, rid_data=rid_data))

# Extract the first block from each piece of html obj.
text_html_obj = []
for i, y in enumerate(extract_html_obj):
    try:
        text_html_obj .append([x.text for x in y.get_first_block()])
    except IndexError:
        pass
    except ValueError:
        pass

for i_url in ref_html_dir:
    res = requests.get(i_url)
    print(res.status_code)

for i_text in A.get_text_by_keyword('Regulation'):
    print(color_back_red(i_text, 'Regulation'))

for i_obj in html_obj:
    A = reg_fe.ExtractRegDoc(*i_obj, rid_data=rid_data)
    for i in A.get_title_structure():
        print(i)
    print('-----------\n\n\n')

import importlib
importlib.reload(proc_text)
B = proc_text.MultiAnalyseDoc(text_html_obj)
B.multi_ngram(3, 10)
B.multi_ner(5)
B.multi_keyword(5)
