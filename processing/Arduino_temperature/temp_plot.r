rm(list= ls())

# =========================================================================== #
# Load lib & location
# =========================================================================== #

# Make sure that we use the local libraries, which is a lot faster.
.libPaths("C:/Users/C35612.LAUNCHER/R-3.4.1/library")

library(stringr)
library(data.table)
library(ggplot2)
library(zoo)

loc_data = "C:\\Users\\C35612.LAUNCHER\\Production_data\\Temperature"
loc_archive = "C:\\Users\\C35612.LAUNCHER\\Production_data\\Temperature\\archive"
loc_plot = "C:\\Users\\C35612.LAUNCHER\\Production_data\\Temperature\\plot"

# Load functions
loc_fun = "C:\\Users\\C35612.LAUNCHER\\Production_code\\Functions"
setwd(loc_fun)
source("generic_sensor_functions.R")
# =========================================================================== #
# Load data
# =========================================================================== #


setwd(loc_data)
last_file = getLatestFile()
temp_data = fread(last_file)

# =========================================================================== #
# Prep data
# =========================================================================== #


temp_data = as.data.table(temp_data )
temp_data[,time := as.POSIXct(paste(day,paste(hour,minute,second,sep = ":")), format="%d %H:%M:%S")]
temp_data[,temperature := ((sensorVal*5/1024)-0.5)*100]
temp_data[,TEMP := rollmean(temperature,10,na.pad = TRUE,align = "right")]
temp_data[,ID := "DIA"]

year_str = sprintf("%.2i",min(temp_data$year))
month_str = sprintf("%.2i",min(temp_data$month))
day_str = sprintf("%.2i",min(temp_data$day))

name_archive = paste0(year_str,month_str,day_str)

temp_data = temp_data[,.(time,TEMP ,ID)] 
temp_data = temp_data[!is.na(TEMP)]

# =========================================================================== #
# Save dataprepsteps
# =========================================================================== #




setwd(loc_archive)
fwrite(temp_data.to_csv(name_archive)


# =========================================================================== #
# Plot data
# =========================================================================== #

setwd(loc_plot)
png("latest_temp_plot.png",width = 900,height = 600,res = 120)
ggplot(data = temp_data,aes(x = time,y = TEMP, colour = ID,alpha = 0.5)) + geom_line(size = 1.2) +
scale_colour_manual(values=c("blue","green")) +
ggtitle("Temperatuur meting DIA-kamer + KNMI temperatuur Schiphol")+
ylab("temperatuur (C)")

dev.off()
