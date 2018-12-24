from TransactionAnalysis import createTable as cT

# Usage cases - Initialize the labels

# Initialize all the labels
A = cT.InitLabels()
# Now write them with the standard name:
# label_dict.txt
A.write_json()

# Read and edit the labels
B = cT.EditLabels('label_dict.txt')
B.get_cat()
B.get_label('city')
B.get_query('city')
B.update_cat({'yourmom': {'test': 'is fat'}})
B.get_cat()
B.get_label('yourmom')
B.get_query('yourmom')

# Undo all the changes that we did...
B.write_json()
C = cT.EditLabels('label_dict.txt')
C.get_cat()
C.remove_query('yourmom', ['test'])
C.remove_cat(['yourmom'])
C.write_json()

