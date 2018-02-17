rm(list = ls())

library(data.table)
library(zoo)
library(ggplot2)

location_fun  = "/RServer/NFShare/Application/Development/Rlab/Global/R/functions/"
location_self = "/home/c35612/"
location_local_data = "//solon.prd/files/P/Global/Users/C35612/Userdata/Documents/1. Data Innovation Analytics/code/assignment 00 - Experiment/assignment 00 - Hive Overview"


setwd(location_fun)
source("connect_function.R")
source("col_handling.R")


# =========================================================================== #
#       MAKE CONNECTIONS
# =========================================================================== #

con_HIVE <- connectHive("")



# =========================================================================== #
#       DEFINE FUNCTIONS
# =========================================================================== #

readTable <- function(i_table,con_HIVE) {
    out <- tryCatch(
        { 
            format_table = as.data.table(dbGetQuery(con_HIVE,paste("describe formatted",i_table)))
			a1 = format_table[grep("Owner",col_name),.(col_name,data_type)]
			a2 = format_table[grep("Create",col_name),.(col_name,data_type)]
			a3 = format_table[grep("Last",col_name),.(col_name,data_type)]
			a4 = format_table[grep("numRows",data_type),.(data_type,comment)]
			a5 = format_table[grep("rawDataSize",data_type),.(data_type,comment)]
			overview = rbindlist(list(a1,a2,a3,a4,a5))
			overview[,table_name := i_table]
			overview[,table_name := i_database]
			overview = dcast(overview,table_name ~ col_name,value.var = "data_type")
        },
        error=function(cond) {
            message(paste("Table does not seem to exist:", i_table))
           # message("Here's the original error message:")
           # message(cond)
            # Choose a return value in case of error
            return(data.table(NA))
        }
    )    
    return(out)
}
	

readDatabase <- function(i_database,con_HIVE) {
    out <- tryCatch(
        {
            dbSendUpdate(con_HIVE,paste("use",i_database))
			table_list = as.data.table(dbGetQuery(con_HIVE,"show tables"))
        },
        error=function(cond) {
            message(paste("Database does not seem to exist:", i_database))
            return(-1)
        }
	)    
return(out)
}


# =========================================================================== #
#       MAKE OVERVIEW
# =========================================================================== #

database_list = as.data.table(dbGetQuery(con_HIVE,"show databases"))

#i_database = database_list[4,]
#i_table = table_list[1,]

counter = 1
final_overview = list()

for(i_database in database_list$database_name){
	cat("* * * * * * * * * * * * * * *\n")
	#cat(i_database,"--------------------------------\n")
	#cat("* * * * * * * * * * * * * * *\n")
	
	table_list = readDatabase(i_database,con_HIVE)
	
	if(class(table_list) == "numeric"){
		next
	}
	
	for(i_table in table_list$tab_name){
		#cat(i_table,"\n")
		#cat("\n")
		overview = readTable(i_table,con_HIVE)
		final_overview[[counter]] = overview
		counter = counter + 1
	}
}

final_final_overview = rbindlist(final_overview,use.names = TRUE,fill = TRUE)

new_names = gsub("\\s+|[[:punct:]]","",names(final_final_overview),perl = TRUE)

setnames(final_final_overview,names(final_final_overview),new_names)

replaceValue(final_final_overview,value_x = "  ",value_y = "",use.gsub = TRUE)
replaceValue(final_final_overview,value_x = " $",value_y = "",use.gsub = TRUE)

final_final_overview = final_final_overview[!is.na(Owner)]
final_final_overview[,year := substr(CreateTime,nchar(CreateTime)-4,nchar(CreateTime))]
final_final_overview[,month := substr(CreateTime,5,8)]
final_final_overview[,day := substr(CreateTime,11,13)]
final_final_overview[,date := paste(year,month,day,sep = "-")]
final_final_overview[,date := gsub("\\s","",date)]
final_final_overview[,date := as.Date(date,format = "%Y-%b-%d")]

final_final_overview[,yearmon := as.yearmon(date)]

final_final_overview[,rawDataSize := as.numeric(rawDataSize)]
final_final_overview[is.na(rawDataSize), rawDataSize := 0]
final_final_overview[,rawDataSize_gb := rawDataSize*10^-9]
final_final_overview[,rawDataSize_gb_month := sum(rawDataSize_gb), by = yearmon]
final_final_overview[,rawDataSize_gb_month_user := sum(rawDataSize_gb), by = .(yearmon,Owner)]

setwd(location_self)
fwrite(final_final_overview,file = "overview_hive.csv",row.names = FALSE)


setwd(location_local_data)
input_data = fread("overview_hive.csv")

input_data[,yearmon := as.yearmon(as.Date(date))]

input_data[,rawDataSize_gb_day := sum(rawDataSize_gb), by = date]

ggplot() + geom_bar(data = input_data,aes(x = date,y = rawDataSize_gb_month_user,group = Owner),stat = "identity")


