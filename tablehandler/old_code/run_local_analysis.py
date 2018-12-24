from TransactionAnalysis.createTagger import *


def mean_list(x):
    return sum(x)/len(x)

raw_dir = r'D:\data\Testing_data\Transaction\raw_transactions'
prep_dir = r'D:\data\Testing_data\Transaction\prep_transactions'
label_dir = r'D:\data\Testing_data\Transaction\label_transactions'

i_year = 2017
test_2017 = HandleFile(i_year, path_in=label_dir)
A = test_2017.load_trx_file()

i_year = 2016
test_2016 = HandleFile(i_year, path_in=prep_dir)
B = test_2016.load_trx_file()

super_expens_2016 = [a.loc[a.supermarkt != ''].VALUE.sum() for a in A]
super_expens_2017 = [a.loc[a.supermarkt != ''].VALUE.sum() for a in B]

mean_list(super_expens_2016)
mean_list(super_expens_2017)