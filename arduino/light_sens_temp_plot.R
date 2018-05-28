#light_sens_plot.R

# =========================================================================== #
# Libraries & Locations
# =========================================================================== #

.libPaths("C:/Users/C35612.LAUNCHER/R-3.4.1/library")
location_data = "C:\\Users\\C35612.LAUNCHER\\Production_data\\Light_sensor_temp"

library(data.table)
library(ggplot2)
library(zoo)
# =========================================================================== #
# Loading data
# =========================================================================== #

setwd(location_data)
details = file.info(list.files(pattern="*.csv"))
details = details[with(details, order(as.POSIXct(mtime))), ]
files = rownames(details)
details = as.data.table(details)
details[,files := files]
details = details[,.(files,atime)]
details = details[!is.na(atime)]
last_file = details[.N]$files

# =========================================================================== #
# Data prep
# =========================================================================== #


light_sens_temp_data = fread(last_file)
light_sens_temp_data[,time := as.POSIXct(paste(day,paste(hour,minute,second,sep = ":")), format="%d %H:%M:%S")]
light_sens_temp_data[,sensorVal_1 := as.numeric(sensorVal_1)]

light_sens_temp_data[,temperature := ((sensorVal*5/1024)-0.5)*100]
light_sens_temp_data[,TEMP := rollmean(temperature,10,na.pad = TRUE,align = "right")]
light_sens_temp_data[,sensorVal_1_mean := rollmean(sensorVal_1,10,na.pad = TRUE,align = "right")]

light_sens_temp_data = light_sens_temp_data[,.(time,TEMP,sensorVal_1_mean)] 
light_sens_temp_data = light_sens_temp_data[!is.na(TEMP)]
light_sens_temp_data = light_sens_temp_data[!is.na(sensorVal_1_mean)]

# =========================================================================== #
# Plot data
# =========================================================================== #


setwd(loc_plot)
png("latest_light_sensor_temp_plot.png",width = 900,height = 600,res = 120)
ggplot() + geom_line(data = light_sens_temp_data,aes(x = time,y = TEMP,alpha = 0.5),size = 1.2) + 
geom_line(data = light_sens_temp_data,aes(x = time,y = sensorVal_1_mean,alpha = 0.5),size = 1.2) +
scale_colour_manual(values=c("blue","green")) +
ggtitle("Temperatuur meting DIA-kamer + Illumination")
dev.off()
