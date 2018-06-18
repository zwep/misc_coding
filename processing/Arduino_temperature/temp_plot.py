
# =========================================================================== #
# Libraries & Locations
# =========================================================================== #

import numpy as np
import pandas as pd
import os
import sys
import pylab
 
loc_data = "C:\\Users\\C35612.LAUNCHER\\Production_data\\Temperature"
loc_archive = "C:\\Users\\C35612.LAUNCHER\\Production_data\\Temperature\\archive"
loc_plot = "C:\\Users\\C35612.LAUNCHER\\Production_data\\Temperature\\plot"

import glob
import os

# =========================================================================== #
# Loading data
# =========================================================================== #

os.chdir(loc_data) 

list_of_files = glob.glob(loc_data+ '\\*.csv') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key = os.path.getctime)
print(latest_file) 
temp_data = pd.read_csv(latest_file)

# =========================================================================== #
# Data prep
# =========================================================================== #
 

temp_data['time'] = pd.to_datetime(temp_data['day'].astype(str) +\
temp_data['hour'].astype(str) + temp_data['minute'].astype(str)  + temp_data['second'].astype(str),format = "%d%H%M%S")
temp_data['temperature'] = ((temp_data['sensorVal']*5/1024)-0.5)*100
temp_data['TEMP'] = temp_data['temperature'].rolling(window = 10).mean()
temp_data['ID'] = "DIA"

name_archive = str(temp_data.year.min()) +  str(temp_data.month.min()).zfill(2) + str(temp_data.day.min()).zfill(2) + "_" +  str(temp_data.day.max()).zfill(2) + ".csv"

choose_col = ['time','TEMP']
temp_data = temp_data[choose_col]
temp_data = temp_data.dropna()

# =========================================================================== #
# Save dataprepsteps
# =========================================================================== #

os.chdir(loc_archive)
temp_data.to_csv(name_archive)

# =========================================================================== #
# Plot data
# =========================================================================== #

os.chdir(loc_plot)
pylab.figure(1)
temp_data = temp_data.set_index('time')
temp_data.plot()
pylab.savefig('latest_temp_plot.png')
