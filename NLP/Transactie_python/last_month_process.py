 
# set libraries
import pandas as pd
import os
import re 
# set locations -----------------------------------------------------------

location_data = "/home/charmmaria/Documents/data/"
location_trx  =  "/home/charmmaria/Documents/data/Transacties/"

# Define constants --------------------------------------------------------

# define date
T_jaar = input("Enter year: \n")
if T_jaar == "": T_jaar = "2016" 
T_maand = input("Enter month: \n")
if T_maand == "": T_maand = "1" 


T_year = T_jaar
T_month = T_maand.zfill(2) 
T_day = "01"
T_date = "/".join([T_year,T_month,T_day])
 
location_data_year = location_trx + T_year

# Define own name
myname = "S\\.D\\. HARREVELT"
# Another special description
rent_string = "HUUR"

# load data  --------------------------------------------------------------
os.chdir(location_trx) 
f_super = open("marktTable.csv")
f_city  = open("cityTable.csv")
f_shop  = open("shopTable.csv")
typeTRX  = pd.read_csv("typeTable.csv")

# Here I fixed the reading
supermarkt = f_super.read()
n_supermarkt = supermarkt.count("\n")
supermarkt = re.sub("\n","|",supermarkt,count = n_supermarkt-1)
supermarkt = re.sub("\n","",supermarkt).upper()

steden     = f_city.read().replace("\n","|").upper() 
shops      = f_shop.read().replace("\n","|").upper()
  
f_super.close()
f_city.close() 
f_shop.close()

S_names       = ["REK_NR", "CUR", "DATE", "DC_IND","VALUE",
                  "TGN_REK_NR", "NM_TGN_REK", "DATE_2", 
                  "MUT_IND", "V1","DESCR_1","DESCR_2"]


# Get the correct file -----------------------------------------------------
os.chdir(location_data_year)
reg_Month = '.*'+T_month+'.txt'
month_file = [f for f in os.listdir('.') if re.match(reg_Month, f)][0]

data_TRX = pd.read_csv(month_file,header = 0) 
data_TRX = data_TRX.fillna("")
data_TRX = data_TRX.iloc[:,0:12]
data_TRX.columns = S_names
 

# Edit columns
data_TRX['DATE_2'] = pd.to_datetime(data_TRX['DATE_2'],format = "%Y%m%d")
data_TRX['DESCR_1'] = data_TRX['DESCR_1'].str.upper() 
data_TRX['NM_TGN_REK'] = data_TRX['NM_TGN_REK'].str.upper() 

 
# Edit second description

reg_string = re.compile("([0-9]{2}:[0-9]{2})")
data_TRX['DESCR_2'] = [reg_string.sub("\\1",x) for x in data_TRX['DESCR_2']]

id_descr2 = data_TRX['DESCR_2'].str.contains(reg_string)
data_TRX.loc[~id_descr2,'DESCR_2'] = "" 

# Edit first description (contains place and store)

# - find city
spec_col = "STAD"

reg_string = re.compile(".*(" + steden[0:(len(steden)-1)] + ").*")
data_TRX[spec_col] = [reg_string.sub("\\1",x) for x in data_TRX['DESCR_1']]
id_descr1 = data_TRX[spec_col].str.contains(reg_string)
data_TRX.loc[~id_descr1,spec_col] = "" 


# - find supermarkets
spec_col = "SUPERMARKT"

reg_string = re.compile(".*(" + supermarkt[0:(len(supermarkt)-1)] + ").*")
data_TRX[spec_col] = [reg_string.sub("\\1",x) for x in data_TRX['DESCR_1']]
id_descr1 = data_TRX[spec_col].str.contains(reg_string)
data_TRX.loc[~id_descr1,spec_col] = "" 
# replace ALBERTHEIJN WITH ALBERT HEIJN

# - find shops
spec_col = "WINKEL"

reg_string = re.compile(".*(" + shops[0:(len(shops)-1)] + ").*")
data_TRX[spec_col] = [reg_string.sub("\\1",x) for x in data_TRX['DESCR_1']]
id_descr1 = data_TRX[spec_col].str.contains(reg_string)
data_TRX.loc[~id_descr1,spec_col] = "" 
 
 
# Sum expenses to myself
# Immediately remove my own transactions
id_credit = data_TRX['DC_IND'] == "C"
id_debit  = data_TRX['DC_IND'] == "D"

# Get own transactions
id_myname = pd.Series([bool(re.search(myname,x)) for x in data_TRX['NM_TGN_REK']])
# Get rent transactions
id_huur   = pd.Series([bool(re.search(rent_string,x)) for x in data_TRX['DESCR_1']])


# These have to happen here, because of index
sparen_totaal = data_TRX.loc[id_debit & id_myname,'VALUE'].sum() 
gift_totaal   = data_TRX.loc[id_credit & id_myname,'VALUE'].sum()
huur_totaal = data_TRX.loc[id_debit & id_huur,'VALUE'].sum() 

# subselect such that we can continue the analysis
#data_TRX = data_TRX[~id_myname]
#data_TRX = data_TRX[~id_huur]


# Get the proper row ids to get the data we want
id_mutind = data_TRX['MUT_IND'] == "ei" 
id_supermarkt = data_TRX['SUPERMARKT'] != "" 
id_winkel = data_TRX['WINKEL'] != "" 


incasso_totaal    = data_TRX.loc[id_mutind,'VALUE'].sum()
supermarkt_totaal = data_TRX.loc[id_supermarkt,'VALUE'].sum()
winkel_totaal     = data_TRX.loc[id_winkel,'VALUE'].sum()
debit_totaal      = data_TRX.loc[id_debit,'VALUE'].sum()
rest_totaal       = debit_totaal - (incasso_totaal + supermarkt_totaal + winkel_totaal + huur_totaal)
credit_totaal     = data_TRX.loc[id_credit,'VALUE'].sum()

supermarkt_group  = data_TRX.loc[id_supermarkt].groupby(['SUPERMARKT'])['VALUE'].sum()
supermarkt_group  = supermarkt_group.reset_index().values.tolist()
 
winkel_group  	  = data_TRX.loc[id_winkel].groupby(['WINKEL'])['VALUE'].sum()
winkel_group      = winkel_group.reset_index().values.tolist()


print(incasso_totaal,supermarkt_totaal,winkel_totaal,debit_totaal,rest_totaal,credit_totaal) 

