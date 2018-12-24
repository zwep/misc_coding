
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, inch, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


# =========================================================================== #
#       DEFINE REPORT
# =========================================================================== #

report_doc = SimpleDocTemplate("target_distribution.pdf", pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30,
                               bottomMargin=18)
report_doc.pagesize = landscape(A4)
report_elements = []
report_table_style = TableStyle([('ALIGN', (1, 1), (-2, -2), 'RIGHT'), ('TEXTCOLOR', (1, 1), (-2, -2), colors.red), 
                                 ('VALIGN', (0, 0), (0, -1), 'TOP'), ('TEXTCOLOR', (0, 0), (0, -1), colors.blue), 
                                 ('ALIGN', (0, -1), (-1, -1), 'CENTER'), ('VALIGN', (0, -1), (-1, -1), 'MIDDLE'), 
                                 ('TEXTCOLOR', (0, -1), (-1, -1), colors.green), ('INNERGRID', (0, 0), (-1, -1),
                                                                                  0.25,  colors.black),
                                 ('BOX', (0, 0), (-1, -1), 0.25, colors.black)])
report_style = getSampleStyleSheet()
report_style = report_style["BodyText"]
report_style.wordWrap = 'CJK'


table_data_2 = [[Paragraph(cell, report_style) for cell in row] for row in info_table_2]
report_table_2 = Table(table_data_2)
report_table_2.setStyle(report_table_style)

report_elements.append(report_table_1)
report_elements.append(report_table_2)

# =========================================================================== #
#       Write report
# =========================================================================== #

os.chdir(location_report)
#Send the data and build the file
report_doc.build(report_elements)
