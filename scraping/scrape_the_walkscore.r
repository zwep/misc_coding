# Should be run on BI Edge Node
location_CREDAB.eenheden = "/dia/c35612/CREDAB/Eenhedendata"
location_CREDAB = "/dia/c35612/CREDAB/"
setwd(location_CREDAB.eenheden)
list_file = list.files()

name_sel  = c("KlantId","ObjectId","FRR","AssetType","EenheidId","Huurder","HuurExpiratieDatum")

# Load data - via merge -------------------------------------------------------
# We start at 6 here.. because before that the data quality is very low
i_file = list_file[6]
month  = gsub("\\D","",i_file)

setwd(location_CREDAB.eenheden)
credab_rbind = unique(fread(i_file))[,Maand := month]

# This can be done better
for(i_file in list_file[-(1:6)]){
# Merge the other kantoren locations to it
month = gsub("\\D","",i_file)
temp  = fread(i_file)[,Maand := month]
setkey(temp,NULL)
temp = unique(temp)
credab_rbind = rbind(credab_rbind,temp, use.names = TRUE, fill = TRUE)}


sel_col_loc = c("Adres","Huisnummer","Toevoeging","Postcode","Plaatsnaam","LandObject","AssetType")
credab_rbind = unique(credab_rbind[,sel_col_loc,with = FALSE])

setwd(location_CREDAB)
fwrite(credab_rbind,file = "credab_locations_scrape.csv",row.names = FALSE)

#----------- over to local
# For the scraping...
# Actually performed this on Python..

.libPaths(.libPaths()[2])
library(rvest) 
library(data.table)

setwd("C:\\Users\\C35612.LAUNCHER\\temp_data")
data_loc = fread("credab_locations_scrape.csv")

data_loc[,Plaatsnaam := gsub("\\s+","",gsub("\\(([a-z].*)\\)","",Plaatsnaam))]

data_loc[,Adres := gsub("\\s+"," ",Adres)]
data_loc[,Adres := gsub(" ","-",Adres)]

data_loc[,Huisnummer := gsub("^([0-9]+).*","\\1",Huisnummer)]
data_loc = data_loc[!is.na(as.numeric(Huisnummer))]
data_loc[,Huisnummer := as.numeric(Huisnummer)]

data_loc[,url_loc := paste(Huisnummer,Adres,Plaatsnaam,sep = "-")]

url_data = "https://www.walkscore.com/score/"

data_loc[,walkscore := ""]

adres_loc_combi = data_loc[,.N,.(Adres,Plaatsnaam)][,N:=NULL]
setkey(data_loc,Adres,Plaatsnaam)
# Selected only te first adres-plaats combination
data_loc_sel = data_loc[.(adres_loc_combi), mult = "first"]

#fwrite(data_loc_sel,file = "credab_locations_sel_scrape.csv",row.names = FALSE)

for(i in 1:10){
	i_data_loc = data_loc_sel[i]$url_loc
	url_send = paste0(url_data,i_data_loc)
	result = tryCatch({
		url_walk_score <- read_html(url_send)
		score_thing = html_nodes(url_walk_score,"div.block-header-badge.score-info-link")
		result = gsub(".*(score.*png).*","\\1",score_thing)}, 
    error = function(e){paste("ERROR:",e)}    # a function that returns NA regardless of what it's passed
	)  
	data_loc_sel[i]$walkscore = result
}

  
# ----  Back to R again

library(data.table)


.libPaths(.libPaths()[2])
library(rvest) 
library(data.table)

setwd("C:\\Users\\C35612.LAUNCHER\\temp_data")
data_loc = fread("credab_locations_scrape.csv")
scrape_score = fread("walkscore_completed.csv")
scrape_score[,walkscore_nr := gsub("score/([0-9]+)\\..*","\\1",walkscore)]

data_loc_walkscore = merge(data_loc,scrape_score[,.(Adres,Postcode,walkscore_nr)],by= c("Adres","Postcode"),all.x = TRUE)
 