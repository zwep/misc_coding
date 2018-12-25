# Using SOM

In this part of the code we explain the algorithm SOM to you - Self Organizing Map. It is a very useful clustering 
algorithm that really trains on your data.

Before you dive into the code... you can start off here to get some general information about the results... In this 
directory there are two files:

* SOM.py - this implements the SOM algorithm and contains comments for almost every step;
* example_SOM.py - this shows two examples using the defined SOM algorithm.

For more information about the SOM algorithm, I would like to point you to internetpages
 
  * [One source of info](http://users.ics.aalto.fi/jhollmen/dippa/node9.html)
  * [Another source of info](http://www.ai-junkie.com/ann/som/som1.html)
  
They are just some quick-to-grab-google search results. But informative nonetheless.
 
 
## Result of the example_SOM.py
##### Showing the first simple example
 
 Understanding the concept of SOM is quite simple to just say... we map higher dimensional-data to a 2-D plane, more 
 or less... To put it even more abstractly: we reduce dimension of the input data and plot them in clusters.
 
 
 Eventhough these concepts are quite abstract, I think it suffies to work with them for this introduction. Let's 
 start with a simple clustering of 15 colors. Having defined these 15 colors in their RGB-format and put them in the 
 SOM (on a 20 by 30 grid), we get the following results after 400 iterations
 
 ![Voila](../plot/Clustering/simple_SOM.png "A simple clustering of colors")
 
 As you can see, we have some nice clustering plus some gradient moving from one color to the other, which is what 
 you would expect. Obtaining the cluster boundaries of the different clusters is a different ball game. To do 
 something like that you need to turn to specific clustering algorithms.
 
 ##### Showing the second more complex example
 
 In the above example we have simple 3-dimensional data, which conveniently plots to RGB values. But usually we 
 don't have data that can be interpreted in that way. That is why I also added an example where you have higher 
 dimensional data. But in order to see WHAT is going on, I still kept the interpretation of the colors.
 
 The first example showed the clustering of 15 different colors, hence this gave as input a 15 x 3 matrix. To modify 
 it to this example, we took a permutation of these 15 colors and appended these to the original color-matrix, such 
 that we have a 15 x 6 matrix. Here we have that the first three columns represent the original 15 colors, while the 
 last three columns are the permuted version of this.
 
 We will train it with the same parameters as the previous SOM. But showing the output will be a bit different this 
 time, we need to slice over each set of three columns in order to graphically show it as colors. And here we see 
 something fun happening. 
  
  Take a look at the first slice
 
 ![Voila](../plot/Clustering/mutli_layer_SOM_0.png "The first layer")
  
 
 Whow! These are the original colors! Amazing. Now lets look at the following two layers
 
 ![Voila](../plot/Clustering/mutli_layer_SOM_1.png "The second layer")
 ![Voila](../plot/Clustering/mutli_layer_SOM_2.png "The third layer")
 
 And what do we se here... yeah well... nothing special. And that's what it makes so special! Especially in contrast 
 with the following layer
 
 ![Voila](../plot/Clustering/mutli_layer_SOM_3.png "The fourth layer")
 
 Tadaa! We see that the SOM clustered the colors in exactly the right way, for both version! How cool is that. Hence 
 the algorithm was able to 'converge' on different level sets.
 
 
 