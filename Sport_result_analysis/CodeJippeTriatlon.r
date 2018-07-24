.libPaths(.libPaths()[2])
location_tri_data = "C:\\Users\\C35612.LAUNCHER\\Testing_data\\Triathlon"
location_tri_data_plots = "C:\\Users\\C35612.LAUNCHER\\Testing_data\\Triathlon_plots"

library(ggplot2)
library(data.table)
library(stringr)
library(ReporteRs)

#name_file = "20170812_DataTriatlonJippe"
name_file = "20170930_DataTriatlonJippe"
setwd(location_tri_data)
#data_tri = readLines("DataTriatlonHans.txt")

data_tri = readLines(paste0(name_file,".txt"))
#data_tri = fread("DataTriatlonHans.txt",sep = "\t)

total_set = list()
counter=  1
for(i in str_split(data_tri,pattern = "\t")){
	if(length(i) == 18) total_set[[counter]] = i; counter = counter+1}


final_table = as.data.table(transpose(total_set))
setnames(final_table,names(final_table),unlist(final_table[1,]))
final_table = final_table[!1,]

# It has trouble here with getting times above one hour
final_table[,PltsTot := as.numeric(PltsTot)]
final_table[,ZwemNum := gsub(":","\\.",substring(Zwem,4,8))]
final_table[,ZwemNum := as.numeric(ZwemNum)]
#final_table[,ZwemNum := as.POSIXlt("01:00:37",format = "%H:%M:%S")]

final_table[,FietsNum := gsub(":","\\.",substring(Fiets,4,8))]
final_table[,FietsNum := as.numeric(FietsNum)]

final_table[,LoopNum := gsub(":","\\.",substring(Loop,4,8))]
final_table[,LoopNum := as.numeric(LoopNum)]


x_Pers_1   = final_table[grep("Jippe.*Groot",Naam)]$PltsTot
Pers_1_Naam = final_table[grep("Jippe.*Groot",Naam)]$Naam

# Line plot without trendline NEW PLOT
myplot = ggplot(data = final_table) + 
geom_line(aes(x = PltsTot, y = ZwemNum, color = "Zwemmen"),size = 0.9)+ geom_smooth(aes(x = PltsTot, y = ZwemNum),alpha = 0.5, color = "black",method = "lm") +
geom_line(aes(x = PltsTot, y = FietsNum, color = "Fietsen" ),size = 0.9)+ geom_smooth(aes(x = PltsTot, y = FietsNum),alpha = 0.5, color = "black",method = "lm") +
geom_line(aes(x = PltsTot, y = LoopNum, color = "Rennen"),size = 0.9)+ geom_smooth(aes(x = PltsTot, y = LoopNum),alpha = 0.5, color = "black",method = "lm") + 
xlim(c(0,200)) + ylim(c(5,50)) + geom_vline(xintercept = x_Pers_1) +
								annotate(c("text","text","text"), 
								label = c(Pers_1_Naam), 
								angle = 0,x = c(x_Pers_1), y = c(45), 
								size = 5, colour = "black") +
								geom_point(data = final_table[PltsTot %in% c(x_Pers_1)],aes(x = PltsTot, y = ZwemNum, color = "Zwemmen")) +
								geom_point(data = final_table[PltsTot %in% c(x_Pers_1)],aes(x = PltsTot, y = FietsNum, color = "Fietsen")) +
								geom_point(data = final_table[PltsTot %in% c(x_Pers_1)],aes(x = PltsTot, y = LoopNum, color = "Rennen")) +
xlab("Totaal plaats") + ylab("Aantal minuten") + ggtitle("Relatie tijden en plaats")


setwd(location_tri_data_plots)
# Line plot with trendline
name_plot = paste0(name_file,".png")
png(name_plot)
ggplot(data = final_table) + 
geom_line(aes(x = PltsTot, y = ZwemNum, color = "Zwemmen"),size = 0.9) + geom_smooth(aes(x = PltsTot, y= ZwemNum),method = "lm") +
geom_line(aes(x = PltsTot, y = FietsNum, color = "Fietsen" ),size = 0.9) + geom_smooth(aes(x = PltsTot, y= FietsNum),method = "lm") +
geom_line(aes(x = PltsTot, y = LoopNum, color = "Rennen"),size = 0.9) +  geom_smooth(aes(x = PltsTot, y= LoopNum),method = "lm") +
xlim(c(0,200)) + ylim(c(5,50)) + geom_vline(xintercept = final_table[grep("Hans Fleer",Naam)]$PltsTot) +
xlab("Totaal plaats") + ylab("Aantal minuten") + ggtitle("Relatie tijden en plaats")
dev.off()

# This right here is super useful!!!
name_pptx = paste0(name_file,".pptx")
mydoc = pptx(  )
mydoc = addSlide( mydoc, slide.layout = "Title and Content" )
mydoc = addTitle( mydoc, "Relatie tijden en plaats" )
mydoc = addPlot( mydoc, function( ) print( myplot ), vector.graphic=TRUE) 
writeDoc( mydoc, file = name_pptx)



# Point plot without trendline
ggplot(data = final_table) + 
geom_point(aes(x = PltsTot, y = ZwemNum, color = "Zwemmen"),size = 0.9)+ 
geom_point(aes(x = PltsTot, y = FietsNum, color = "Fietsen" ),size = 0.9) +
geom_point(aes(x = PltsTot, y = LoopNum, color = "Rennen"),size = 0.9) + 
xlim(c(0,200)) + ylim(c(5,50)) + geom_vline(xintercept = final_table[grep("Hans Fleer",Naam)]$PltsTot) + 
xlab("Totaal plaats") + ylab("Aantal minuten") + ggtitle("Relatie tijden en plaats")


# Point plot with trendline
ggplot(data = final_table) + 
geom_point(aes(x = PltsTot, y = ZwemNum, color = "Zwemmen"),size = 0.9)+ geom_smooth(aes(x = PltsTot, y= ZwemNum),method = "lm") +
geom_point(aes(x = PltsTot, y = FietsNum, color = "Fietsen" ),size = 0.9) + geom_smooth(aes(x = PltsTot, y= FietsNum),method = "lm") +
geom_point(aes(x = PltsTot, y = LoopNum, color = "Rennen"),size = 0.9) + geom_smooth(aes(x = PltsTot, y= LoopNum),method = "lm") +
xlim(c(0,200)) + ylim(c(5,50)) + geom_vline(xintercept = final_table[grep("Hans Fleer",Naam)]$PltsTot) + 
xlab("Totaal plaats") + ylab("Aantal minuten") + ggtitle("Relatie tijden en plaats")

