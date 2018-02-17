// Creating subset of terminal ID for Amsterdam
create table di_temp.seb_amsterdam_termid 
	as select * from termid_pc_mcc_view where term_postcode 
		rlike "1000|1001|1002|1003|1005|1006|1007|1008|1009|1011|1012|1013|1014|1015|1016|1017|1018|1019|1020|1021|1022|1023|1024|1025|1026|1027|1028|1030|1031|1032|1033|1034|1035|1036|1037|1040|1041|1042|1043|1044|1045|1046|1047|1051|1052|1053|1054|1055|1056|1057|1058|1059|1060|1061|1062|1063|1064|1065|1066|1067|1068|1069|1070|1071|1072|1073|1074|1075|1076|1077|1078|1079|1080|1081|1082|1083|1086|1087|1090|1091|1092|1093|1094|1095|1096|1097|1098|1099" 
		and term_country = 'NL';

// Create a subset of the transaction set for the month december		
// Join with terminal id table

create table di_temp.seb_amsterdam_2015 as select trim(substr(regexp_extract(OMS_REGEL_1,'NR:(.+)',1),1,9)) as term_id_trx, 
verwerking_datum, substr(lpad(verwerking_tijd,8,"0"),1,2) as hour, count(*) as count_total, sum(mutatie_bedrag_eur) as sum_value  
from di_aed.v_geld_contract_event  where year = '2015' and mutatie_soort_code = '426' and  source_ind = 'D' and verwerking_datum >= '2015-01-01'   
group by trim(substr(regexp_extract(OMS_REGEL_1,'NR:(.+)',1),1,9)), verwerking_datum, substr(lpad(verwerking_tijd,8,"0"),1,2);


create table di_temp.seb_amsterdam_201512 as 
	select trim(substr(regexp_extract(OMS_REGEL_1,'NR:(.+)',1),1,9)) as term_id_trx, 
	verwerking_datum, substr(lpad(verwerking_tijd,8,"0"),1,2) as hour, count(*) as count_total, 
	sum(mutatie_bedrag_eur) as sum_value 
		from di_aed.v_geld_contract_event 
			where year = '2015' and mutatie_soort_code = '426' and 
			source_ind = 'D' and verwerking_datum > '2015-11-30' 
				group by trim(substr(regexp_extract(OMS_REGEL_1,'NR:(.+)',1),1,9)), verwerking_datum, substr(lpad(verwerking_tijd,8,"0"),1,2);
			

create table di_temp.seb_amsterdam_2015_termid as 
select * from di_temp.seb_amsterdam_2015 Z join di_temp.seb_amsterdam_termid  Y  ON Z.term_id_trx = Y.TERM_ID where Y.TERM_POSTCODE != '0000000000'  AND Y.TERM_POSTCODE != '00000000' AND Y.TERM_POSTCODE != '' AND SUBSTRING(Y.TERM_POSTCODE,1,4) > '1000';

			
create table di_temp.seb_amsterdam_201512_termid as 
select * from di_temp.seb_amsterdam_201512 Z join di_temp.seb_amsterdam_termid  Y  ON Z.term_id_trx = Y.TERM_ID where Y.TERM_POSTCODE != '0000000000'  AND Y.TERM_POSTCODE != '00000000' AND Y.TERM_POSTCODE != '' AND SUBSTRING(Y.TERM_POSTCODE,1,4) > '1000';

// Now we have transaction ready..
// Off to R on the Rserver


options(java.parameters = "-Xmx8g")

library(data.table)



connectHive<- function(database,username = "c35612", password = "")
{
  library("DBI")
  library("rJava")
  library("RJDBC")
  .jinit("/home/rdevlop/hive-jdbc-2.1.0.2.5.0.0-1245.jar")
  for(l in list.files('/home/rdevlop/check')){ .jaddClassPath(paste("/home/rdevlop/check",l,sep=""))}
  for(l in list.files('/home/rdevlop/hadoop/lib')){ .jaddClassPath(paste("/home/rdevlop/hadoop/lib/",l,sep=""))}
  for(l in list.files('/home/rdevlop/hadoop/conf')){.jaddClassPath(paste("/home/rdevlop/hadoop/conf",l,sep=""))}
  for(l in list.files('/home/rdevlop/hadoop/client')){.jaddClassPath(paste("/home/rdevlop/hadoop/client/",l,sep=""))}
  for(l in list.files('/home/rdevlop/hive2/lib')){ .jaddClassPath(paste("/home/rdevlop/hive2/lib/",l,sep=""))}
  #print()

  
  
  drv <- JDBC("org.apache.hive.jdbc.HiveDriver", .jclassPath(), identifier.quote="'")
  conn <- dbConnect(drv, "jdbc:hive2://infetrmstn-003-p1.hdp.nl.eu.abnamro.com:2181,infetrmstn-002-p1.hdp.nl.eu.abnamro.com:2181,infetrmstn-001-p1.hdp.nl.eu.abnamro.com:2181/;serviceDiscoveryMode=zooKeeper;zooKeeperNamespace=hiveserver2", username,password)
  #conn <- dbConnect(drv, "jdbc:hive2://10.30.4.43:8787/", username,password)
  
  return(conn)
}


con_HIVE = connectHive("di_temp")

db_name = "seb_amsterdam_2015_termid"
#db_name = "seb_amsterdam_201512_termid"
query  = "select * from	 default.seb_amsterdam_december_trx_aggr_pc" 
query  = paste0("select * from	 di_temp.", db_name)

z = as.data.table(dbGetQuery(con_HIVE,query))

n_ding = nchar(db_name) + 1
n_names = names(z)
setnames(z,n_names,substr(n_names,n_ding,nchar(n_names)))
# setnames(z,n_names,gsub('\\.','',n_names))  # In case the point is still there

z = z[term_id != "00000000"]
z = z[trx_datum_min < verwerking_datum & verwerking_datum < trx_datum_max]

z[,term_postcode := toupper(gsub(' ','',term_postcode))]
z[grep("^NL",term_postcode), term_postcode := gsub("^NL","",term_postcode)]
z[nchar(term_postcode) == 7, term_postcode := substr(term_postcode,1,6)]

#z[,total_count := sum(count_total),by = .(substr(term_postcode,1,4),verwerking_datum,substr(verwerking_tijd_new,1,2))]
z[,total_count := sum(count_total),by = .(substr(term_postcode,1,4), verwerking_datum, hour)]
#z[,total_value := sum(`_c4`),by = .(substr(term_postcode,1,4),,verwerking_datum,substr(verwerking_tijd_new,1,2))]
z[,total_value := sum(sum_value), by = .(substr(term_postcode,1,4),verwerking_datum,hour)]
result = unique(z[,.(verwerking_datum, hour, post = substr(term_postcode,1,4),total_count,total_value)])

data_loc = "/RServer/NFShare/Application/Development/Rlab/"
setwd(data_loc)

# fwrite(z,file = "amsterdam_aggr_201512.csv",row.names = FALSE)
fwrite(z,file = "amsterdam_aggr_2015.csv",row.names = FALSE)


# ================================= #
# LOCAL R!!
# ================================= #

location_data = "//solon.prd/branches/P/Global/Users/C35612/Userdata/Documents/1. Data Innovation Analytics/Data/HackatonAmsterdam"
location_data = "D:\\data\\trx_gemeente"
location_SHP = "//solon.prd/branches/P/Global/Users/C35612/Userdata/Documents/1. Data Innovation Analytics/Data/SHP_csv"
\\solon.prd\files\P\Global\Users\C35612\Userdata\Documents/1. Data Innovation Analytics/Data/SHP_csv
# ------------- #				# ------------- #
# ------------ READ THE POLYGONS  ------------- #
# ------------- #				# ------------- #

.libPaths(.libPaths()[2])

library(data.table)
library(ggplot2)
library(mapview)
library(sp)
library(htmlwidgets)
library(ReporteRs)

# Read in the shape files
setwd(location_SHP)
#sel_col = c("long","lat","PC4","id","order")
region_positions = fread("POSTCODE_amsterdam.csv")

setkey(region_positions,PC4,order)
# Subselect only necessary values
# Split the long-lat coords into a list with IDs
lonlat_pc = region_positions[,.(long,lat,PC4)]
lonlat_pc_list <- split(lonlat_pc[,.(long,lat)], lonlat_pc$PC4)
# Make a list of Polygon elements, needed to call Polygons
poly_list <- lapply(lonlat_pc_list, Polygon)
# Create a Polygonsss for the list of Polygon items
poly_list_s <- lapply(seq_along(poly_list), function(i) Polygons(list(poly_list[[i]]), ID = names(poly_list)[i]  ))
# Now we can make a Spatial Polygon, by adding the projection
poly_list_sp <- SpatialPolygons(poly_list_s, proj4string = CRS("+proj=longlat +datum=WGS84") )
# Here we create the DataFrame, we need to match the row.names to the IDs...
poly_df_pc <- SpatialPolygonsDataFrame(poly_list_sp, data.frame(post = unique(lonlat_pc$PC4), row.names = unique(lonlat_pc$PC4)))


# ------------- #				# ------------- #
# ------------ READ THE FEATURES	------------- #
# ------------- #				# ------------- #		
			
# Read in the data that we want to plot
setwd(location_data)
# feature_CMP = fread("amsterdam_aggr_2015_dec.csv")
# feature_CMP = fread("amsterdam_aggr_201512.csv")
feature_CMP = fread("amsterdam_aggr_2015.csv")
 
feature_CMP[,date:= as.POSIXct(date,format = "%Y-%m-%dT%H")]
result = feature_CMP

seq_time = as.POSIXct(paste0("2015-01-",sprintf("%.2i",1:31),' 01:00:00'),format = "%Y-%m-%d %H")

myplot = ggplot(result) + geom_line(aes(x=date,y=total_count, color=sel_post)) + scale_x_datetime(breaks=seq_time, labels = as.Date(seq_time)) + theme(axis.text.x = element_text(angle=45))
#ggplot(result) + geom_line(aes(x=date,y=norm_count, color=sel_post)) + scale_x_datetime(breaks=seq_time, labels = as.Date(seq_time)) + theme(axis.text.x = element_text(angle=45))
 
name_file = 'amsterdam_plot_pres' 
# This right here is super useful!!!
name_pptx = paste0(name_file,".pptx")
mydoc = pptx(  )
mydoc = addSlide( mydoc, slide.layout = "Title and Content" )
mydoc = addTitle( mydoc, "Overzicht aantal transacties over 2015-12" )
mydoc = addPlot( mydoc, function( ) print( myplot ), vector.graphic=TRUE) 
writeDoc( mydoc, file = name_pptx)
 
# --- stupid plot --- #
 
sel_date = "2015-08-01" 
sel_hour = "12:00"
sel_POSIX = as.POSIXct(paste(sel_date,sel_hour))
feature_CMP_sel = feature_CMP[date == sel_POSIX]

poly_df_pc <- SpatialPolygonsDataFrame(poly_list_sp, data.frame(post = unique(lonlat_pc$PC4), row.names = unique(lonlat_pc$PC4)))
poly_df_pc@data = merge(poly_df_pc@data,feature_CMP_sel,by.x = "post",by.y = "sel_post",all.x=TRUE)

poly_df_pc = merge(poly_df_pc,feature_CMP_sel,by.x = "post",by.y = "sel_post",all.x=TRUE)

ggplot(poly_df_pc, aes(x=long, y=lat, group=group))+geom_polygon()
  geom_polygon(aes(fill=total_count))+
  geom_path(colour="grey50")+
  scale_fill_gradientn("2012 Marriages",
                       colours=rev(brewer.pal(8,"Spectral")), 
                       trans="log", 
                       breaks=c(100,300,1000,3000,10000))+
  theme(axis.text=element_blank(), 
        axis.ticks=element_blank(), 
        axis.title=element_blank())+
  coord_fixed()
  
id_NA = is.na(poly_df_pc@data$total_count)
poly_df_pc@data$total_count[id_NA] = 0
poly_df_pc@data$total_value[id_NA] = 0

poly_df_pc@data[is.na(poly_df_pc@data)] = "1950-01-01 01"

			
# Plot the polygons
mapview(poly_df_pc,zcol = "total_count", legend = TRUE)
mapview(poly_df_pc,zcol = "norm_count", legend = TRUE)
mapview(poly_df_pc,col.regions = mapviewGetOption("raster.palette")(256))
	
feature_CMP_sel[,total_count := as.factor(total_count)]	
mapview(poly_df_pc,zcol = "total_count", legend = TRUE, at = seq(0,1,0.01))

# Loops over all the posibiliteis and then plots them
seq_time = paste0("2015-12-",sprintf("%.2i",1:31)) 
for(i_time in seq_time){
	poly_df_pc <- SpatialPolygonsDataFrame(poly_list_sp, data.frame(post = unique(lonlat_pc$PC4), row.names = unique(lonlat_pc$PC4)))

	# Select a month to visualize the data
	#T_month = "2015-12-01" 
	T_month = i_time
	feature_CMP_sel = feature_CMP[verwerking_datum == T_month]
	feature_CMP_sel[,norm_count := total_count/max(total_count)]
	# ------------- #				# ------------- #
	# ------------ ADD THE FEATURES	------------- #
	# ------------- #				# ------------- #		
				 
	# Add the plot data to the Spatial Polygon
	poly_df_pc@data = merge(poly_df_pc@data,feature_CMP_sel,by.x = "post",by.y = "post")
	poly_df_pc@data[is.na(poly_df_pc@data)] = 0

				
	# Plot the polygons
	mapview(poly_df_pc,zcol = "norm_count", legend = TRUE)
}


# Maybe move to Python
