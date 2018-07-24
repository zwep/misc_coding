location_tri_data = "C:\\Users\\C35612.LAUNCHER\\Testing_data\\Triathlon"

library(ggplot2)
library(data.table)
library(stringr)
library(ReporteRs)

setwd(location_tri_data)
data_tri = readLines("DataTriatlonHans.txt")
#data_tri = fread("DataTriatlonHans.txt",sep = "\t)

total_set = list()
counter=  1
for(i in str_split(data_tri,pattern = "\t")){
	if(length(i) == 18) total_set[[counter]] = i; counter = counter+1}


final_table = as.data.table(transpose(total_set))
setnames(final_table,names(final_table),unlist(final_table[1,]))
final_table = final_table[!1,]

final_table[,PltsTot := as.numeric(PltsTot)]
final_table[,ZwemNum := gsub(":","\\.",substring(Zwem,4,8))]
final_table[,ZwemNum := as.numeric(ZwemNum)]

final_table[,FietsNum := gsub(":","\\.",substring(Fiets,4,8))]
final_table[,FietsNum := as.numeric(FietsNum)]

final_table[,LoopNum := gsub(":","\\.",substring(Loop,4,8))]
final_table[,LoopNum := as.numeric(LoopNum)]


x_Hans   = final_table[grep("Hans Fleer",Naam)]$PltsTot
Hans_Naam = final_table[grep("Hans Fleer",Naam)]$Naam
x_Wout = final_table[grep("Wouter Nij",Naam)]$PltsTot
Wout_Naam = final_table[grep("Wouter Nij",Naam)]$Naam
x_Jurjen   = final_table[grep("Jurjen We",Naam)]$PltsTot
Jurjen_Naam = final_table[grep("Jurjen We",Naam)]$Naam

# Line plot without trendline NEW PLOT
myplot = ggplot(data = final_table) + 
geom_line(aes(x = PltsTot, y = ZwemNum, color = "Zwemmen"),size = 0.9)+ geom_smooth(aes(x = PltsTot, y = ZwemNum),alpha = 0.5, color = "black",method = "lm") +
geom_line(aes(x = PltsTot, y = FietsNum, color = "Fietsen" ),size = 0.9)+ geom_smooth(aes(x = PltsTot, y = FietsNum),alpha = 0.5, color = "black",method = "lm") +
geom_line(aes(x = PltsTot, y = LoopNum, color = "Rennen"),size = 0.9)+ geom_smooth(aes(x = PltsTot, y = LoopNum),alpha = 0.5, color = "black",method = "lm") + 
xlim(c(0,200)) + ylim(c(5,50)) + geom_vline(xintercept = x_Hans) +
								geom_vline(xintercept = x_Jurjen) +
								geom_vline(xintercept = x_Wout) +
								annotate(c("text","text","text"), 
								label = c(Hans_Naam,Wout_Naam,Jurjen_Naam), 
								angle = 0,x = c(x_Hans,x_Wout,x_Jurjen), y = c(45,45,45), 
								size = 5, colour = "black") +
								geom_point(data = final_table[PltsTot %in% c(x_Hans,x_Jurjen,x_Wout)],aes(x = PltsTot, y = ZwemNum, color = "Zwemmen")) +
								geom_point(data = final_table[PltsTot %in% c(x_Hans,x_Jurjen,x_Wout)],aes(x = PltsTot, y = FietsNum, color = "Fietsen")) +
								geom_point(data = final_table[PltsTot %in% c(x_Hans,x_Jurjen,x_Wout)],aes(x = PltsTot, y = LoopNum, color = "Rennen")) +
xlab("Totaal plaats") + ylab("Aantal minuten") + ggtitle("Relatie tijden en plaats")

# Line plot with trendline
png("ResultaatTriatlon.png")
ggplot(data = final_table) + 
geom_line(aes(x = PltsTot, y = ZwemNum, color = "Zwemmen"),size = 0.9) + geom_smooth(aes(x = PltsTot, y= ZwemNum),method = "lm") +
geom_line(aes(x = PltsTot, y = FietsNum, color = "Fietsen" ),size = 0.9) + geom_smooth(aes(x = PltsTot, y= FietsNum),method = "lm") +
geom_line(aes(x = PltsTot, y = LoopNum, color = "Rennen"),size = 0.9) +  geom_smooth(aes(x = PltsTot, y= LoopNum),method = "lm") +
xlim(c(0,200)) + ylim(c(5,50)) + geom_vline(xintercept = final_table[grep("Hans Fleer",Naam)]$PltsTot) +
xlab("Totaal plaats") + ylab("Aantal minuten") + ggtitle("Relatie tijden en plaats")
dev.off()

# This right here is super useful!!!
mydoc = pptx(  )
mydoc = addSlide( mydoc, slide.layout = "Title and Content" )
mydoc = addTitle( mydoc, "Relatie tijden en plaats" )
mydoc = addPlot( mydoc, function( ) print( myplot ), vector.graphic=TRUE) 
writeDoc( mydoc, file = "HansTriatlon.pptx" )



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



# Werkte niet

ggplot(data = final_table) + 
geom_bar(aes(x = PltsTot, y = ZwemNum, fill = "Zwemmen"),position = "dodge",stat = "identity",size = 0.9) +
geom_bar(aes(x = PltsTot, y = FietsNum, fill= "Fietsen" ),position = "dodge",alpha = 0.5,stat = "identity",size = 0.9) +
geom_bar(aes(x = PltsTot, y = LoopNum, fill= "Rennen"),position = "dodge",alpha = 0.5,stat = "identity",size = 0.9) 

+ 


xlim(c(0,200)) + ylim(c(5,50)) + 
xlab("Totaal plaats") + ylab("Aantal minuten") + ggtitle("Relatie tijden en plaats")
