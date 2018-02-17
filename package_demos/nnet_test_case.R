## This is a test script
library(nnet)
library(data.table)
library(rpart)


# Learnings...
# Found the function max.col
# which gives the maximum POSITION per column 
# Can also be used on data.tables!

# It is a more general approach to which.max() for vectors

# Further, the function class.ind can give a nice 
# frequency plot for a given vector
# ex.
set.seed(1)
test = sample(letters[1:5],100,replace = TRUE)
test2 = class.ind(test)
test3 = colSums(test2)
 
# Investigate base package and stats package and methods package further
N_base_prev = 30
N_base_max  = N_base_prev + 10
ls("package:base")[N_base_prev:N_base_max]
# Useful function to check which packages/DLLs are loaded
getLoadedDLLs()

# The operator @ is concerned with classes... (S4 classes to be precise)
# Recall that 'numeric', 'integer', etc. are all clases
# To define a new class, call
setClass()
# To create a new class, call
new()
# To get the class, call
getClass()
# To get the names and types, call
getSlots()
# To get the names, call
slotNames()

# Being redirected to the function slot()
# Getting the following example code
setClass("track", representation(x="numeric", y="numeric"))
myTrack <- new("track", x = -4:4, y = exp(-4:4))


# How to check which classes are made?
#setClass
#net
#str
#




# Check what is in the package nnet
data(iris)

# Getting the data
ir = rbind(iris3[,,1],iris3[,,2],iris3[,,3])
targets_full = c(rep("s", 50), rep("c", 50), rep("v", 50))
targets = class.ind(targets_full)
ir_total = as.data.table(cbind(ir,targets_full))

# Making a sample
set.seed(1)
samp = c(sample(1:50,25), sample(51:100,25), sample(101:150,25))

# Building a model
ir1 = nnet(ir[samp,], targets[samp,], size = 2, rang = 0.1,
           decay = 5e-4, maxit = 200)

# rpart is not working well somehow..
# it is taking a long time and then it crashes
#rpart(targets_full ~ ., data = ir_total[samp,], method = "class")
# I would like to compare the rpart results with nnet
# (And also logit or others)

# Function for displ
test.cl <- function(true, pred) {
  true <- max.col(true)
  cres <- max.col(pred)
  table(true, cres)
}
test.cl(targets[-samp,],predict(ir1, ir[-samp,]))
