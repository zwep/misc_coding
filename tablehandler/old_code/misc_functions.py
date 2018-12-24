

"""
I think I want  a function that is able to do a group by sum over a time period...

Time period being something like years... months.. days...
Grouped by labels... like supermarket, shops, cities, c/d, eic etc.

so we need atleast three parameters
- the data
- the type of period (years, months, days, hours...)
- the thing we want to know the VALUE of
"""

"""
But I also want to compare two lines with eachother.

for example... Credit expenses and debit.. or cumsum and value.. or multiple things
like supermarkt1 and supermarkt2 things

"""

"""
I also want to incorporate an interactive labeling system

And maybe use the JSON format for the processed transactions

path = r'D:\\data\\Testing_data\\Transaction'
A = np.genfromtxt(path + r'\\transactions_2016_08.txt', delimiter=',', dtype='str')
A = pd.DataFrame(A)
B = pd.DataFrame(A).head()
C = json.loads(B.to_json(orient='records'))
os.chdir

"""

"""

# again... from Scratch... what do we want.

# overview of account transactions
# but what...
# - balance over time


# function/script for labeling data...

# Load data for time period

# Create plots..

# Sum over period...

# Visualize
# category totals - summary_table
# category per month - overview_table
# sub category totals
# sub catgeory per month

"""

"""
From this point on we assume prepped data
"""

# What we need... IDs of rows that locate stuf..
# Plot the chosen IDs

# otherway arround...
# What do we want to plot at first?

# Balance over set of months... with a dynamic window please.
# Amount spent on supermarkets...
# Amount spent on cities...
# Amount spent on shops..

# lets keep it that way

"""
 Misc functions
"""


def get_time_interval(n_range):
	"""
	Asks you for a date, returns a date interval
	:param n_range: size of date interval in months
	:return: date range of size n_range
	"""
	# define date
	t_jaar = input("Enter year: \n")
	if t_jaar == "":
		t_jaar = "2016"

	t_maand = input("Enter month: \n")
	if t_maand == "":
		t_maand = "1"

	t_start_date = pd.to_datetime(t_jaar + t_maand.zfill(2) + "01", format="%Y%m%d")
	t_end_date = t_start_date - pd.DateOffset(months=n_range)
	print("Entered", t_start_date, "until ", t_end_date)
	return t_start_date, t_end_date


def load_transaction_time(t_start_date, n_range, col_names):
	"""
	Fucntion to load a set of transaction tables according to the date range
	:param t_start_date:
	:param n_range:
	:param col_names:
	:return:
	"""
	print("Loading transaction data...")
	file_names = np.array(glob.glob("*.txt"))

	reg_month = str(t_start_date.year) + "_" + str(t_start_date.month).zfill(2)

	name_file = np.array([x for x in file_names if bool(re.search(reg_month,x))][0])
	index_file = np.where(file_names == name_file)[0][0]

	if index_file-n_range < 0:
		n_range = 0

	index_range = np.array(np.arange(index_file-n_range, index_file+1))
	file_list = file_names[index_range]
	data_trx = pd.concat((pd.read_csv(f, header=None) for f in file_list))
	data_trx = data_trx.reset_index(drop=True)
	data_trx = data_trx.fillna("")
	data_trx = data_trx.iloc[:, 0:12]
	data_trx.columns = col_names
	print("...loaded")

	return data_trx


def check_cd(data, row_index):
	"""
	derp
	:param data:
	:param row_index:
	:return:
	"""
	cd_count = data.loc[row_index, 'DC_IND'].value_counts()
	cd_count = cd_count.index.values
	if len(cd_count) != 1:
		cd_count = ['0']
	return cd_count[0]


def graph_id(id_list, id_legend):
	derp = []
	aa = pd.DataFrame(id_list)

	for a in aa:
		b = np.nonzero(aa[a])
		n = b[0].shape[0]
		if n > 0:
			c = np.array(np.meshgrid(b, b)).reshape(2, n*n)
			d1 = tuple(map(tuple, c.transpose()))
			derp.append(d1)

	d1 = tuple(map(tuple, np.concatenate(derp)))
	d2 = list(set(d1))

	return d2


def plot_trx_graph(trx_con, id_legend):
	g = nx.Graph()
	g.add_nodes_from(np.unique(trx_con))
	g.add_edges_from(trx_con)
	labels = dict((n, id_legend[n]) for n, d in g.nodes(data=True))
	nx.draw(g, labels=labels)
	plt.show()


def to_df(x, id_legend, cd_list):
	y = pd.DataFrame(x)
	y['name'] = id_legend
	y['dc_ind'] = cd_list
	y = y.set_index(['name', 'dc_ind'])
	y = y.fillna(0)
	return y


def get_own_trx(data, my_name='S\\.D\\. HARREVELT'):
	# Transactions from myself
	id_myname = data['NM_TGN_REK'].str.contains(my_name, regex=True)
	return id_myname


def get_trx_id(data_trx):
	# This one is nt good
	id_credit = data_trx['DC_IND'] == "C"
	id_debit = data_trx['DC_IND'] == "D"

	# Traveling train transactions
	id_trein = data_trx['NM_TGN_REK'].str.contains("^NS GROEP", regex=True)
	id_treinD = id_debit & id_trein
	id_treinC = id_credit & id_trein

	# Rent
	id_huur = id_debit  & data_trx['DESCR_1'].str.contains(" HUUR ", regex=True)

	# Several standard transaction codes
	id_declaC = data_trx['MUT_IND'] == "cb"
	id_ei = data_trx['MUT_IND'] == "ei"
	id_sb = data_trx['MUT_IND'] == "sb"

	# Self created labels
	id_mrkt   = data_trx['SUPERMARKT'] != ""
	id_winkel = data_trx['WINKEL'] != ""
	id_stad   = data_trx['STAD'] != ""

	# Concat all id's
	id_list = list([id_sparen, id_gift, id_huur, id_ei, id_sb, id_declaC, id_treinD, id_treinC, id_mrkt, id_winkel,
	                id_stad])

	# Check the missing
	id_missing = pd.DataFrame(id_list).sum(axis=0)
	id_missing = id_missing == 0

	# Add missing
	id_list.append(id_missing)
	id_list.append(id_credit)
	id_list.append(id_debit)

	id_legend = list(['sparen', 'gift', 'huur', 'incasso', 'salaris', 'crediet_decla', 'ns', 'ns_vergoeding',
	                  'supermarkt', 'winkel', 'stad', 'rest', 'crediet', 'debit'])

	id_dict = dict(zip(id_legend, range(len(id_legend))))
	return id_list, id_legend, id_dict


# Aggregate over different levels
def trx_aggregate(data_trx, id_type, i_type=""):
	# This one sucks as well
	if i_type != "":
		agg_total = data_trx.loc[id_type].groupby([i_type])['VALUE'].sum()
		agg_year = data_trx.loc[id_type].groupby([i_type, 'year'])['VALUE'].sum()
		agg_month = data_trx.loc[id_type].groupby([i_type, 'yearmonth'])['VALUE'].sum()
		agg_day = data_trx.loc[id_type].groupby([i_type, 'DATE'])['VALUE'].sum()
	else:
		agg_total = data_trx.loc[id_type]['VALUE'].sum()
		agg_year = data_trx.loc[id_type].groupby('year')['VALUE'].sum()
		agg_month = data_trx.loc[id_type].groupby('yearmonth')['VALUE'].sum()
		agg_day = data_trx.loc[id_type].groupby('DATE')['VALUE'].sum()

	return agg_total, agg_year, agg_month, agg_day


def lol_plot(data_trx, id_list, id_mrkt, id_winkel, id_stad, id_legend):
	# ----------------------------------------------------------------------------#
	# check identifiers
	# ----------------------------------------------------------------------------#
	cd_list = [check_cd(x) for x in id_list]

	# TODO I could also perform a sum() operation here
	# A sum opertion as in... sum every part and check the (sub)totals
	# Then the sub totals should be equal in a way
	# ----------------------------------------------------------------------------#
	# Create aggregation tables
	# ----------------------------------------------------------------------------#

	total_agg = []
	year_agg = []
	month_agg = []
	day_agg = []

	for i_index, i_name in enumerate(id_legend):
		i_total, i_year, i_month, i_day = trx_aggregate(id_list[i_index])
		total_agg.append(i_total)
		year_agg.append(i_year)
		month_agg.append(i_month)
		day_agg.append(i_day)

	dc_color_dict = {'D': 'r', 'C': 'g', '0': 'k'}

	my_colors = [(x/15.0, x/20.0, 0.75) for x in range(len(id_legend))]

	total_agg = to_df(total_agg)
	total_agg.columns = ['total_value']

	year_agg = to_df(year_agg)
	year_agg = year_agg.transpose().unstack()
	year_agg.columns = ['name','dc_ind','date','value']
	month_agg = to_df(month_agg)
	month_agg = month_agg.transpose().unstack()
	month_agg.columns = ['name', 'dc_ind', 'date', 'value']
	day_agg = to_df(day_agg)

	sub_mrkt_total,sub_mrkt_year,sub_mrkt_month,sub_mrkt_day = trx_aggregate(id_mrkt,'SUPERMARKT')
	sub_winkel_total,sub_winkel_year,sub_winkel_month,sub_winkel_day = trx_aggregate(id_winkel,'WINKEL')
	sub_stad_total,sub_stad_year,sub_stad_month,sub_stad_day = trx_aggregate(id_stad,'STAD')

	print("Overview of all the balances that are computed")

	my_dc_color = list(map(dc_color_dict.get,total_agg.reset_index().dc_ind))
	# --------
	# PLot totals per (sub)category
	# --------
	# 4 plots
	total_agg.reset_index().plot.bar(x = 'name', y = 'total_value')
	sub_mrkt_total.reset_index().plot.bar(x = 'SUPERMARKT', y = 'VALUE')
	sub_winkel_total.reset_index().plot.bar(x = 'WINKEL', y = 'VALUE')
	sub_stad_total.reset_index().plot.bar(x = 'STAD', y = 'VALUE')

	# --------
	# PLot year per (sub)category
	# --------
	# 4 plots
	year_agg.reset_index().pivot(index = 'date',columns = 'name', values = 'value').plot.bar()
	sub_mrkt_year.reset_index().pivot(index = 'year',columns = 'SUPERMARKT', values = 'VALUE').plot.bar()
	sub_winkel_year.reset_index().pivot(index = 'year',columns = 'WINKEL', values = 'VALUE').plot.bar()
	sub_stad_year.reset_index().pivot(index = 'year',columns = 'STAD', values = 'VALUE').plot.bar()

	# --------
	# PLot months per (sub)category
	# --------
	# 4 plots
	# Months...

	month_agg.reset_index().pivot(index = 'date',columns = 'name', values = 'value').plot(style= '.-')
	sub_mrkt_month.reset_index().pivot(index = 'yearmonth',columns = 'SUPERMARKT', values = 'VALUE').plot(style= '.-')
	sub_winkel_month.reset_index().pivot(index = 'yearmonth',columns = 'WINKEL', values = 'VALUE').plot(style= '.-')
	sub_stad_month.reset_index().pivot(index = 'yearmonth',columns = 'STAD', values = 'VALUE').plot(style= '.-')
	plt.show()
	# --------
	# PLot days per (sub)category
	# --------
	# Make date column
	A = day_agg.transpose().unstack().reset_index()
	A.columns = ['name','dc_ind','date','value']
	A.date = pd.to_datetime(A.date, format = '%Y%m%d')
	A['month'] = [x.month for x in A.date]

	for a,grp in A.groupby('month'):
		grp.pivot(index = 'date',columns = 'name', values = 'value').plot(style = '.-')

		# can these also be plotted per group?
	a = sub_mrkt_day.reset_index()
	a.DATE = pd.to_datetime(a.DATE, format = '%Y%m%d')
	a.pivot(index = 'DATE',columns = 'SUPERMARKT', values = 'VALUE').plot(style = '.-')

	b = sub_winkel_day.reset_index()
	b.DATE = pd.to_datetime(a.DATE, format = '%Y%m%d')
	b.pivot(index = 'DATE',columns = 'WINKEL', values = 'VALUE').plot(style = '.-')

	c = sub_stad_day.reset_index()
	c.DATE = pd.to_datetime(a.DATE, format = '%Y%m%d')
	c.pivot(index = 'DATE',columns = 'STAD', values = 'VALUE').plot(style = '.-')



# ZIjn alle ids mutually exlusive?? Denk t tniet heh
# Zo kan je kijken wat bij elkaar hoort en wat niet
# A = pd.DataFrame([id_gift,id_sb,id_declaC,id_treinC]).sum()
# plt.plot(id_list[0])
# plt.show()



# from matplotlib_venn import venn3


# v = venn3(subsets=(1,1,0,1,0,0,0))
# v.get_label_by_id('100').set_text('First')
# v.get_label_by_id('010').set_text('Second')
# v.get_label_by_id('001').set_text('Third')
# plt.title("Not a Venn diagram")
# plt.show()

# ----------------------------------------------------------------------------#
# show examples of top 5
# ----------------------------------------------------------------------------#

# col_example = ['NM_TGN_REK','DATE','DESCR_1','VALUE']
# data_trx.loc[id_incasso,col_example].sort_values('VALUE').tail()
# data_trx.loc[id_debit,col_example].sort_values('VALUE').tail()
# data_trx.loc[id_credit,col_example].sort_values('VALUE').tail()

# Rest
# data_trx.loc[~(id_treinreis | id_treinreis_vergoeding | id_salaris | id_sparen | id_crediet_decla | id_gift | id_huur | id_incasso | id_mrkt | id_winkel | id_stad)].sort_values('VALUE').tail()

# Wat is cb qua mitind?
# En je moet de NS voor reizen er nog uit alen