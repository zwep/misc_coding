# Run the creation of plots


import TransactionAnalysis.createTagger as cTag

raw_dir = r'D:\data\Testing_data\Transaction\raw_transactions'
prep_dir = r'D:\data\Testing_data\Transaction\prep_transactions'
label_dir = r'D:\data\Testing_data\Transaction\label_transactions'


i_year = 2017
i_month = 9
test = cTag.HandleFile(i_year, month=i_month, path_in=prep_dir)

labeling_class = cT.EditLabels('label_dict.txt', path=label_dict_dir)
temp_label_files = cTag.LabelData(i_year, path_in=prep_dir, labels=labeling_class, path_out=label_dir)

A = test.load_trx_file()
A_sub = []
for A_i in A:
    pass
    # t1 = A_i.loc[(A_i.VALUE < -40) & (A_i.VALUE > -90), ['VALUE', 'DATE', 'DESCR_1']]
	# t1 = A_i.loc[(A_i.VALUE < -40) & (A_i.VALUE > -90)]
    # A_sub.append(t1)
    # print(t1)
