# encoding: utf-8

"""

In this file we present some techniques that were used to validate our assumption of the data. It is sometimes
usefull to keep these pieces of code to be sure that it works on every piece of data
"""

# <editor-fold desc='Import libraries'>
import re
import unicodedata

from project.RegulatoryOffice.ridfunctions import get_RID
from project.RegulatoryOffice.ridfunctions import get_html_obj
from project.RegulatoryOffice.fileexplorer import ExtractRegDoc
import geneticalgorithm.optimizer as helper_opt
from helper.miscfunction import diff_list, color_back_red_index, color_back_red
import importlib

# </editor-fold>

# <editor-fold desc='Defining locations and loading data'>
dir_data = r'D:\temp'
dir_reg_doc = dir_data + r'\reg_doc'
dir_reg_html = dir_reg_doc + r'\html'
name_data = 'Export_RID_16012018.xlsx'

rid_data = get_RID(name_data, dir_reg_doc)  # Gives back a dictionaru
html_obj = get_html_obj(dir_reg_html)  # Gives back a tuple of (content, name)
# </editor-fold>

""" a consecutive block of doc-ti titles are only present at the start of the document """

# <editor-fold>

# Or actually... the doc-ti distri. Where are the 'clustesr' of doc-ti
test_list = []
for i_html in html_obj:
    A = ExtractRegDoc(*i_html, rid_data=rid_data)
    B = A.get_title_structure()
    print('\n---\n', A.name)
    A_docti = [x for x in B if x[1] == 'doc-ti']
    test_list.append(diff_list([x[0] for x in A_docti]))

# From this we see that indeed a block of one's is only present at the start
for i_numbers in test_list:
    print(i_numbers)

# </editor-fold>

""" assuming that Beautifulsoup.find_all(class_=True) will preserve order """

# <editor-fold>
# The only way to test is to go throuh this one
# A = ExtractRegDoc(*html_obj[0], rid_data=rid_data).get_title_structure()
# And compare it with this one...
# html_obj[0][0].text

# </editor-fold>

""" Sometimes the ANNEX of a document is not part of the html, because its a picture """

# <editor-fold>
# Check out
# http://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32017R2382&from=EN
# D:/temp/reg_doc/html/CELEX_32017R2382.html

# </editor-fold>

""" Beautifulsoup.find_all(class_=True) will contain better formatted information than  Beautifulsoup.find_all(
class_=False)"""

# <editor-fold>
A = ExtractRegDoc(*html_obj[0], rid_data=rid_data)
test_sentence = 'he liabilities referred to in the first subparagraph, were they not excluded from bail-in, can be s'
test_true = [x.text for x in A.input_class_true if re.findall(test_sentence, x.text)]
test_false = [x.text for x in A.input_class_false if re.findall(test_sentence, x.text)]


def print_content_red(input_text, input_sentence, name):
    """ Printing some things red in a list of texts... """
    for i_index, i_text in enumerate(input_text):
        print('\n\n----****---- ', name, '----****---- length: ', len(input_text), ' item: ', i_index)
        print(color_back_red(i_text, input_sentence))


print_content_red(test_true, test_sentence, 'class_true')
print_content_red(test_false, test_sentence, 'class_false')

# From this we see that the `false' class contains the full text of the whole document (since that part has not been
# assigned a class). This also gives away some of the inner workings of find_all

# </editor-fold>

""" Does Beautifulsoup.find_all(class_=True) cover all the text that we need? """

# <editor-fold>
A = ExtractRegDoc(*html_obj[0], rid_data=rid_data)

test_sentence = unicodedata.normalize('NFKD', [x.text for x in A.input_class_true][1])
test_class_true = ' '.join([x.text for x in A.input_class_true])

test_1 = re.sub('\s+', ' ', re.sub('\n+', '', unicodedata.normalize('NFKD', test_class_true)))
test_2 = re.sub('\s+', ' ', re.sub('\n+', ' ', unicodedata.normalize('NFKD', A.input_html.text)))


def match_string_func(input_distr, input_text_1, input_text_2):
    """
    Here we can compare text_1 with text_2. As in.. we go over each character/element in text_1 and compare it with
    text_2

    :param input_distr:
    :param input_text_1:
    :param input_text_2:
    :return:
    """
    import math

    mapping_score = 0
    mapping_list = [(-1, -1)]
    # input_distr = [int(x) for x in input_distr]

    for x_origin, x_char in enumerate(input_text_1):
        start_point = int(mapping_list[-1][1]+1)
        end_point = int(math.ceil(start_point + input_distr[x_origin]))
        for y_origin, y_char in enumerate(input_text_2[start_point:end_point]):  # Give some area
            if x_char == y_char:
                mapping_score += 1
                mapping_list.append((x_origin, y_origin + mapping_list[-1][1] + 1))
                break

    return mapping_score, mapping_list


# Generate initial distribution
importlib.reload(helper_opt)
sub_test_1 = test_1[0:800]
sub_test_2 = test_2[0:800]

B = helper_opt.GeneticAlgorithm(sub_test_1, sub_test_2, max_search=100, n_mut_repeat=20, mut_chance=80, n_pop=100,
                                max_gen=5000)
result = B.run()

tolerance_distr = result[0]
correct_score, correct_list = B.score_individual(tolerance_distr)
x_index_list, y_index_list = zip(*correct_list)

print(color_back_red_index(sub_test_1, list(x_index_list)[1:]))
print('')
print(color_back_red_index(sub_test_2, list(y_index_list)[1:]))

# </editor-fold>

"""" Do all the documents posses an Article 1? """

# Still need to write a script.