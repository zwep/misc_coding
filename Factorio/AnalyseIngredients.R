 rm(list = ls())
location_data = "/home/charmmaria/Documents/data/Factorio"
library(dbscan)
library(igraph)
library(data.table)

setwd(location_data)

data = fread("Ingredients_Factorio.csv")
data[,file := NULL]

cast_data = dcast(data,name_recipe ~ ingredients, value.var = "amount")

for(i in 1:ncol(cast_data)){
  old_col = cast_data[[i]]
  new_col = old_col
  new_col[is.na(new_col)] = 0L 
  
  set(cast_data,j = i, value = new_col)
}

cast_data_ind = copy(cast_data)
for(i in 2:ncol(cast_data_ind)){
  old_col = cast_data_ind[[i]]
  new_col = old_col
  new_col[new_col >0] = 1L
  
  set(cast_data_ind,j = i, value = new_col)
}

cluster_cast_data =  dbscan(cast_data[,-"name_recipe"],eps = 1,minPts = 2)$cluster
cluster_cast_data_ind =  dbscan(cast_data_ind[,-"name_recipe"],eps = 1,minPts = 2)$cluster

View(cbind(cluster_cast_data,cast_data))
View(cbind(cluster_cast_data_ind,cast_data)) 

ckmeans_cast_data_ind = kmeans(cast_data_ind[,-"name_recipe"],10)$cluster
View(cbind(ckmeans_cast_data_ind,cast_data)) 

data_graph= graph_from_data_frame(data, directed = TRUE, vertices = NULL)

setwd(location_data)
png(
  "test.png", 
  res       = 1200
)
plot(data_graph,layout = layout.fruchterman.reingold,
     vertex.size = 1,rescale = FALSE, ylim=c(0,20),xlim=c(0,20), asp = 0)
dev.off()

# Try to use TSNE to reduce dimensionality
library(tsne)
A_clust = tsne(cast_data[,-'name_recipe'])
plot(A_clust)
text(A_clust, labels=cast_data$name_recipe)

A_clust_names = cast_data[,.(name_recipe)]
A_clust_names$x = A_clust[,1]
A_clust_names$y = A_clust[,2]
 
fwrite(A_clust_names, file = 'data_clust.csv', row.name = FALSE)

library(dbscan)
#And then try to cluster htem with dbscan?

#Fuckign dbscan, is super annoying to determine the parameters...
#Here is some python code to visualize the plot

from matplotlib.pyplot import figure, show
import numpy as np
from numpy.random import rand


import os
import pandas as pd
dir_data = "/home/charmmaria/Documents/data/Factorio"

os.chdir(dir_data)
A_clust = pd.read_csv('data_clust.csv')
print_name = A_clust.name_recipe.values
x = A_clust.x.values
y = A_clust.y.values

def onpick3(event):
	ind = event.ind
	print('onpick3 scatter:', ind, np.take(x, ind), np.take(y, ind), np.take(print_name, ind))



fig = figure()
ax1 = fig.add_subplot(111)
col = ax1.scatter(x,y, picker=True) 
fig.canvas.mpl_connect('pick_event', onpick3)

show()