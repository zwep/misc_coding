rm(list = ls())

location_recipe = "/home/charmmaria/.local/share/Steam/steamapps/common/Factorio/data/base/prototypes/recipe"
location_data = "/home/charmmaria/Documents/data/Factorio"

library(data.table)
 
setwd(location_recipe)

result = list()
counter = 1
files = list.files(pattern = ".lua")

for(i_file in files){ 
  cat("------------------------",i_file,"\n")
  z = readLines(i_file)
 
  #result = data.table(name_recipe = character(),ingredients = character(), amount = character())
  distinct_rows = data.table(a = grep("\"recipe\"",z), b = c(grep("\"recipe\"",z)[-1]-1,length(z)))
  for(i in 1:nrow(distinct_rows)){
    i_a = distinct_rows[i,a]
    i_b = distinct_rows[i,b]
    one_recipe = z[i_a:i_b]
    name_recipe = gsub("\\s+|name =|,|\"","",grep("name =",one_recipe,value = TRUE))[1]
    cat(name_recipe,i,"\n")
    if(any(grepl("expensive",one_recipe))){
      one_recipe = one_recipe[grep("normal",one_recipe):grep("expensive",one_recipe)]
    }
    rec_a = grep("ingredients", one_recipe)
    rec_b = grep("result =|results=|result=|results =", one_recipe)
    ingredients = grep("\\d",(gsub(",|\\s+|\\{|\\}","",one_recipe[(rec_a+1):(rec_b-1)])),value = TRUE)
    if(length(ingredients) == 0) ingredients = gsub("\\s+|ingredients =","",one_recipe[(rec_a):(rec_a)])
    dummy = data.table(name_recipe,ingredients)
    dummy[,amount := gsub("\"(.*)\"([0-9]+$)","\\2",ingredients)]
    dummy[,ingredients :=  gsub("\"(.*)\"([0-9]+$)","\\1",ingredients)]
    dummy[,file := i_file]
    result[[counter]] = dummy
    counter = counter + 1
    }
}


for(i in 1:length(result)) result[[i]]$name_recipe = as.character(result[[i]]$name_recipe)
ingredient_table = rbindlist(result)

ingredient_table[grep("name",ingredients), amount := gsub(".*amount=","",amount)]
ingredient_table[grep("name",ingredients), ingredients := gsub(".*name=\"(.*)\".*","\\1",ingredients)]

ingredient_table[grepl("ingredients",ingredients)  & !grepl("\"",ingredients)  , amount :=gsub("ingredients=","",amount)]
ingredient_table[grepl("ingredients",ingredients)  & !grepl("\"",ingredients)  , ingredients :=gsub("ingredients=","",ingredients)]


z = ingredient_table[grepl("ingredients",ingredients) ]
z[,ingredients := paste0(ingredients,"\"",gsub("ingredients=","",amount),"\"")]
z[, ingredients :=gsub("ingredients=","",ingredients)]
z_result = list()
for( i in 1:nrow(z)){
name_test = z[i,name_recipe]
file_test = z[i,file]
test = str_split(z[i,ingredients],pattern = "\"")[[1]]
test = test[test != ""] 
dum_test = data.table(name_recipe = name_test, ingredients = test[seq(1,length(test),2)], amount = test[seq(2,length(test),2)], file = file_test)
z_result[[i]] = dum_test
}
ingredient_table_sub = rbindlist(z_result)

ingredient_table= ingredient_table[!grepl("ingredients",ingredients) ]
ingredient_table = rbind(ingredient_table,ingredient_table_sub,use.names = TRUE)

ingredient_table = ingredient_table[!grep("result",ingredients)]
ingredient_table = ingredient_table[!grep("energy_req",ingredients)] 


ingredient_table[is.na(as.numeric(amount)), amount := "0"]
ingredient_table[,amount := as.numeric(amount)]


setwd(location_data)
fwrite(ingredient_table,file = "Ingredients_Factorio.csv",row.names = FALSE)
