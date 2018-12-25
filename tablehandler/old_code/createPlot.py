# Make a class that can plot the loaded data
# We assume that we are dealing with labeled data

import pandas as pd

from TransactionAnalysis.createTagger import HandleFile
import matplotlib.pyplot as plt


class PlotData(HandleFile):
    """
    Plot stuff
    """
    def __init__(self, year, path_in, month='', path_out=''):
        """
        Upon initialization, creates the prepped files from file_list
        :param year: see HandleFile
        :param path_in: see HandleFile
        :param month: see HandleFile
        :param path_out: see HandleFile
        """
        super().__init__(year, path_in, month, path_out)
        self.data = self.load_trx_file(header=0)
        self.temp_data = ''

    def table_aggregate(self, group_col, list_agg_dict, time_frame):
        """
        :param group_col: column names with which we want to group-by
        :param list_agg_dict: list with items as {colname} : {agg_action}
        :param time_frame: column name indicating time...
        :return:
        """

        agg_data = []
        group_col = group_col + time_frame

        for i_data in self.data:
            x_agg = i_data.groupby(group_col).agg(list_agg_dict)
            agg_data.append(x_agg)

        return agg_data

    def plot_aggregate(self, group_col, list_agg_dict, time_frame):
        """
        :param group_col: column names with which we want to group-by
        :param list_agg_dict: list with items as {colname} : {agg_action}
        :param time_frame: column name indicating time...
        :return:
        """

        test = self.table_aggregate(group_col, list_agg_dict, time_frame)
        test.plot()


raw_dir = r'D:\data\Testing_data\Transaction\raw_transactions'
prep_dir = r'D:\data\Testing_data\Transaction\prep_transactions'
label_dir = r'D:\data\Testing_data\Transaction\label_transactions'

test = []
i_group = ['city']
i_time = ['month']
agg_dict = {'city': 'size', 'VALUE': 'mean'}

A = test.table_aggregate(i_group, agg_dict, i_time)
B = pd.concat(A)
B.columns = ['city_size', 'value_mean']
B = B.reset_index()
B.groupby('city')['value_mean'].plot(legend=True)
plt.show()

# Did I make profit this month?
# Expenses to supermarket by time line.. day, week, motn
# Expenses to supermarkt by type... (Ah, Jumbo, ...)

for i in ['city', 'shop', 'supermarkt']:
    print(test.data[0].groupby(i).agg({i: 'size', 'VALUE': 'mean'}))
    len(test.data[0])
    print(test.data[0].groupby('DC_IND').agg({'DC_IND': 'size', 'VALUE': 'mean'}))

test.data[0].groupby(['DC_IND', 'DATE_2']).agg({'DC_IND': 'size', 'VALUE': 'mean'})

A = test.data[0]
i_col = ['city', 'month']

derp = {'city': 'size', 'month': 'size', 'VALUE': 'mean'}
A.groupby(['city', 'month']).agg(derp)
print(A.groupby(i_col).agg({i_col: 'size', 'VALUE': 'mean'}))

def col_count_mean_value(A, i_col):
    print(A.groupby(i_col).agg({i_col: 'size', 'VALUE': 'mean'}))


def col_count_mean_value(A, i_col):
    print(A.groupby(i_col).agg({i_col: 'size', 'VALUE': 'mean'}))


def get_max_spend(A, n_max, col='VALUE', asc=True):
    """

    :param A:
    :param n_max:
    :param col: can also be CUM_VALUE
    :param asc: max or min
    :return:
    """
    A.sort_values(col, ascending=asc)[0:n_max]

import matplotlib.pyplot as plt

n_plot = len(test.data)
for i in np.arange(1, n_plot):
    plt.subplot(1, n_plot, i)
    derp = test.data[i]['CUM_VALUE']
    if i == 1:
        derp_prev = 0
    else:
        derp_prev = test.data[i-1]['CUM_VALUE'].values[-1]
    derp = derp + derp_prev
    derp.plot(figsize=(30, 7), ylim=(-1500,3000))

plt.show()


plt.subplot(1, 2, 2)



test.data[0].groupby(['city']).agg({'city': 'size', 'VALUE': 'mean'})
