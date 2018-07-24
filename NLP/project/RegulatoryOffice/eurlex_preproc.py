# encoding: utf-8

"""
In this file we preprocess the RID database, such that the content can easily be indexed over folders.

This means that we have a way of downloading those files here... and subsequently order them in the appropriate
folder...

The reason for this is to pre-process the files such to index them, and make search functions easier. The
discriminants in which each file can fall varies... but includes
- Impacted functions
- Impacted business lines
- Existing projects (like MiFID, CRR, GDPR, etc.)
- etc.

It is totally possible that one file can be found in multiple location. The whole reason for this setup is that we
have, in a later stage, easier access to the files through Kivy.
Search function -> relocates the current path to the specified search -> Access the files that were asked.

"""

import project.RegulatoryOffice.ridfunctions as reg_rid
import textprocessing.processtext as text_proc
import project.RegulatoryOffice.fileexplorer as reg_fe
import glob
import importlib
import os
import re

# <editor-fold desc='Defining directories and loading RID data'>
dir_data = r'D:\temp'
dir_reg_doc = dir_data + r'\reg_doc'
src_path = dir_reg_doc + r'\html'
dest_path = dir_reg_doc + r'\html_indexed'

name_data = 'Export_RID_16012018.xlsx'
rid_data = reg_rid.get_RID(name_data, dir_reg_doc)  # Gives back a dictionary
# </editor-fold>

# <editor-fold desc='Here we will download the necessary files from EURLEX using RID'>
a = reg_rid.get_html_url_from_RID(rid_data)
query_eurlex = ' '.join(a)
html_res = reg_rid.call_eurlex_api(query_eurlex, username='test', password='test')
reg_rid.save_html_object(html_res, path=os.getcwd())
# </editor-fold>

# <editor-fold desc='Here we will copy the files to their respective location'>
# Check if this is needed... maybe it is good enough to be able to search through their properties
files_to_index = glob.glob(os.path.join(src_path, '*.html'))
# TODO Here we need to change the class.. and the appropriate actions that need after that.
A = reg_rid.ExtractRidData(files_to_index, rid_data)

# Filemanager is shit. Will keep the code for some time now... but I dont think it is THAT usefull. Fun to make though
for k, v in A.prep_rid_data.items():
    print('***********')
    for i in v.values():
        print('-----', k, i)

# </editor-fold>

# <editor-fold desc='Start up to create the knowledge base'>
# Re-initialize the regdic preprocess step

dir_file = glob.glob(src_path + '\\*html')
import importlib
importlib.reload(reg_fe)
A_JSON = reg_fe.KnowledgeGraph(dir_reg_doc, dir_file=dir_file, file_ext='.html')

# </editor-fold>

# <editor-fold desc='Fill the content for the json dump'>
# This step can only be completed after downloading all the documents..
# TODO Think about downloading refernece odcuments as well..
files_to_index = glob.glob(os.path.join(src_path, '*.html'))
A_RID = reg_rid.ExtractRidData(files_to_index, rid_data)
html_obj = reg_rid.load_html_obj(src_path)  # Gives back a tuple of (content, name)

res_dict = {}
# TODO reduce the runtime (or something) from this command.
for i, i_obj in enumerate(html_obj):
    if i % int(len(html_obj)*0.1) == 0:
        print(int(i*100/len(html_obj)))
    A = reg_fe.ExtractRegDoc(*i_obj, rid_data=rid_data)
    # class_true can give double pieces of text.. which is annoying..
    input_text = [re.sub(r'\n+', '', x[2]) for x in A.text_class_true if re.search('normal', x[1], re.IGNORECASE)]
    input_text = [x for x in input_text if len(x) > 4]

    text_prep = text_proc.CleanText(input_text, language='english')
    text_lem_stop = text_prep.remove_stopwords_text().lemmatize_text()

    # Object to analyse the prepped text
    text_anl = text_proc.AnalyseDoc(text_prep.text)
    text_anl_lem_stop = text_proc.AnalyseDoc(text_lem_stop.text)

    # B = text_proc.AnalyseDoc(input_text)

    try:
        x_keywords = text_anl_lem_stop.most_common_keywords(10)
    except IndexError:
        print('Index {0} displayed an IndexError for Keywords'.format(i))
        x_keywords = text_anl_lem_stop.most_common_keywords(1)
    x_ner = text_anl_lem_stop.most_common_ner(10)
    x_trigram = text_anl_lem_stop.most_common_ngrams(3, 10)

    temp = {i_obj[1]: {'Keywords': [x[0] for x in x_keywords],
                       'NER': [x[0][0] for x in x_ner],
                       'Trigram': [' '.join(x[0]) for x in x_trigram],
                       'Summary': text_anl.get_summary(),
                       'Reference regulation': A.get_reference_regulation(),
                       'Reference directive': A.get_reference_directives()}}
    res_dict.update(temp)

# Now store everything in one big dumpster
final_dict = {}
celex_id_1 = set(A_RID.prep_rid_data.keys())
celex_id_2 = set(res_dict.keys())
celex_id_1.update(celex_id_2)

for z_id in celex_id_1:
    if z_id in A_RID.prep_rid_data.keys():
        final_dict.update({z_id: A_RID.prep_rid_data[z_id]})
    if z_id in res_dict.keys():
        final_dict.update({z_id: res_dict[z_id]})

A_JSON.update_json(final_dict)
A_JSON.write_db_json()

# </editor-fold>

# <editor-fold desc='Use the json dump to look up information..?'>
files_to_index = glob.glob(os.path.join(src_path, '*.html'))
A_JSON = reg_fe.KnowledgeGraph(dir_reg_doc)
# A_RID = reg_rid.ExtractRidData(files_to_index, rid_data)
# B = EurLexMultiAnalyseDoc(r'D:\temp\reg_doc', keyword_things)

# <editor-fold>