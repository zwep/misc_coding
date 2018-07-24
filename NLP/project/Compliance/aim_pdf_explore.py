# encoding: utf-8

"""
Tada
"""


# Compliance stuff
from PyPDF2 import PdfFileReader
import os

# dir_data = r'D:\temp\reg_doc\aim\101_21_51\Links'
dir_data = r'D:\temp'
os.chdir(dir_data)
A = PdfFileReader(open('CELEX_32016R1450.pdf', 'rb'))
for i in dir(A): print(i)


A.documentInfo
A.getXmpMetadata().pdf_keywords
A.getXmpMetadata().dc_description
A.getXmpMetadata().dc_title

A.numPages
A.getPage(5).extractText()

A.getOutlines()  # This gives bookmarks.. but what are the pages refering to the bookmark?
A.resolvedObjects  # NO idea what this is


# Check this one as well
# https://stackoverflow.com/questions/8329748/how-to-get-bookmarks-page-number

# IndirectObject

# encoding:utf-8

"""

Here we write a function to pdf
"""

import os
import PyPDF2

dir_data = r'D:\temp\reg_doc'


def get_pdf_content(path):
    """

    :param path:
    :return:
    """
    content = ""
    num_pages = 1

    # PATH OF THE PDF FILE
    p = open(path, "rb")
    pdf = PyPDF2.PdfFileReader(p)
    for i in range(0, num_pages):
        content += pdf.getPage(i).extractText() + "\n"
    # x = 'testing something \xa0\xa0'
    # y = unicodedata.normalize('NFKD', x)

    content = " ".join(content.replace(u"\xa0", " ").strip().split())

    return content


# Call function
os.chdir(dir_data)
all_pdf_files = os.listdir('.')
one_file = all_pdf_files[0]

all_text = []
for i, x in enumerate(all_pdf_files):
    try:
        all_text.append(get_pdf_content(x))
    except:  # Jup, very bad
        print("nooooo")

# GOd dangit, this one is not helping...
all_text = [get_pdf_content(x).encode("ascii", "ignore") for x in all_pdf_files]
