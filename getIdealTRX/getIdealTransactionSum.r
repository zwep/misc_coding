# Tha query

library(data.table)
source("/RServer/NFShare/Application/Development/Rlab/Global/R/functions/connect_function.R")
con_hive = connectHive()
query_to_di_aed = "select verwerking_datum, mutatie_bedrag_dc_ind, sum(mutatie_bedrag_eur) from di_aed.v_geld_contract_event where year = 2017 and month = 'jul-aug' and (mutatie_soort_code = 477 or mutatie_soort_code = 944) group by mutatie_bedrag_dc_ind,verwerking_datum" 

query_to_di_aed_hour = "select verwerking_datum,hour(from_unixtime(verwerking_tijd)) as time, mutatie_bedrag_dc_ind, sum(mutatie_bedrag_eur) as value from di_aed.v_geld_contract_event where year = 2017 and month = 'jul-aug' and (mutatie_soort_code = 477 or mutatie_soort_code = 944) group by mutatie_bedrag_dc_ind,verwerking_datum,hour(from_unixtime(verwerking_tijd))" 


data_result = data.table(dbGetQuery(con_hive,query_to_di_aed))
data_result = data.table(dbGetQuery(con_hive,query_to_di_aed_hour))

fwrite(data_result,file = "iDealOnline_time.csv",row.names = FALSE)


# ---- to local

loc_data = "\\\\solon.prd\\files\\P\\Global\\Users\\C35612\\UserData\\Desktop\\temp_data"
setwd(loc_data)

library(data.table)
library(ggplot2)
library(RColorBrewer)
data_result = fread("iDealOnline.csv" )
setnames(data_result,names(data_result),c("date","dc","value"))
data_result[,dc := as.factor(dc)]


myColors <- brewer.pal(3,"Set1")[1:2][2:1]
names(myColors) <- levels(data_result$dc)
colScale <- scale_colour_manual(name = "dc",values = myColors)


ggplot(data = data_result, aes(x = date,y = value,fill = dc)) + 
geom_bar(stat = "identity" ) + scale_fill_manual(name = "dc",values = myColors) + 
theme(axis.text.x = element_text(angle = 45, hjust = 1))


#-- hourly data

loc_data = "\\\\solon.prd\\files\\P\\Global\\Users\\C35612\\UserData\\Desktop\\temp_data"
setwd(loc_data)

library(data.table)
library(ggplot2)

data_result = fread("iDealOnline_time.csv" )
setnames(data_result,names(data_result),c("date","time","dc","value"))
data_result[,dc := as.factor(dc)]

data_result[,datetime:= as.POSIXct(paste(date,time,sep = " "),format = "%Y-%m-%d %H")]

myColors <- brewer.pal(3,"Set1")[1:2][2:1]

ggplot(data = data_result[date > '2017-07-27'], aes(x = datetime,y = value,fill = dc)) + 
geom_bar(stat = "identity" ) + scale_fill_manual(name = "dc",values = myColors) + 
theme(axis.text.x = element_text(angle = 45, hjust = 1))

