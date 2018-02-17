#---
import copy
import glob


import pandas as pd
import numpy as np

import io

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, inch, landscape,letter
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak

# =========================================================================== #
#       DEFINE REPORT
# =========================================================================== #

	
report_doc = SimpleDocTemplate("test_report.pdf", pagesize=A4, rightMargin=30,leftMargin=30, topMargin=30,bottomMargin=18)
#report_doc = SimpleDocTemplate("test_report.pdf", pagesize=A4)
report_doc.pagesize = landscape(A4)
report_elements = []
report_table_style = TableStyle([('ALIGN',(1,1),(-2,-2),'RIGHT'),('TEXTCOLOR',(1,1),(-2,-2),colors.red),
                                 ('VALIGN',(0,0),(0,-1),'TOP'),('TEXTCOLOR',(0,0),(0,-1),colors.blue),
                                 ('ALIGN',(0,-1),(-1,-1),'CENTER'),('VALIGN',(0,-1),(-1,-1),'MIDDLE'),
                                 ('TEXTCOLOR',(0,-1),(-1,-1),colors.green),('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
                                 ('BOX', (0,0), (-1,-1), 0.25, colors.black)])
report_style = getSampleStyleSheet()
report_style_normal = report_style["Normal"]
report_style_normal.wordWrap = 'LTR'

report_style_body = report_style["BodyText"]
report_style_body.wordWrap = 'CJK'


# =========================================================================== #
#       DEFINE DATA
# =========================================================================== #

x = np.arange(0,10,1)
y = np.arange(0,1,0.1)
myDataFrame = pd.DataFrame(dict(time=x,perc_10=y,perc_20 = y*2,perc_30 = y*0.5))
myDataFrame1 = copy.deepcopy(myDataFrame)

myDataFrame['variable'] = 'F1' 
myDataFrame1['variable'] = 'F2' 

total_df = pd.concat([myDataFrame,myDataFrame1])
total_df = total_df.set_index('variable')


# =========================================================================== #
#       SAVE PLOTS
# =========================================================================== #
 
for i_level in np.unique(total_df.index):
	test = total_df.loc[i_level]
	test.plot(x = 'time', y = ['perc_10','perc_20','perc_30'],figsize = (10,7))
	plt.axes().fill_between(test.time,test.perc_30,test.perc_20,alpha = 0.5)
	plot_name = "test" + i_level + ".png" 
	plt.savefig(plot_name)


# =========================================================================== #
#       CREATE CONTENT
# =========================================================================== #


# Create the table content
info_table_1 = np.array(myDataFrame1.reset_index())
# Make sure that the content is string to put it in the report
for i in range(0,info_table_1.shape[0]):
    info_table_1[i,:] = [str("{:.3e}".format(x)) if isinstance(x,float) else str(x) for x in info_table_1[i,:]]

info_table_1 = np.insert(info_table_1,0,myDataFrame1.reset_index().columns,axis = 0)
table_data_1 = [[Paragraph(cell, report_style_body) for cell in row] for row in info_table_1]
report_table_1 = Table(table_data_1)
report_table_1.setStyle(report_table_style)

# =========================================================================== #
#       ADD CONTENT
# =========================================================================== #
	
Story=[]

list_images = glob.glob("test*png")
for logo in list_images:
	logo_file = "./" + logo
	im = Image(logo_file,width=10*inch,height=7*inch,kind='proportional')
	Story.append(im)
	Story.append(PageBreak())
 
ptext = '<font size=12>Some text je <br /> \n moeder isdik</font>' 
Story.append(Paragraph(ptext, report_style_normal))

# ptext = '<font size=12>Some text je \
# moeder isdik</font>' 
# Story.append(Paragraph(ptext, report_style_normal))

# ptext = '''
# <seq>. </seq>Some Text<br/>
# <seq>. </seq>Some more test Text
# '''
# Story.append(Paragraph(ptext, report_style["Bullet"]))

# ptext='<bullet>&bull;</bullet>Some Text'
# Story.append(Paragraph(ptext, report_style["Bullet"]))

Story.append(Spacer(0,0.25*inch))

Story.append(report_table_1)
 
report_doc.build(Story)


