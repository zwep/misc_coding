This directory explain the word2vec model to you... or to put it more technical: the skip-gram, cbow and 
character-n-gram versions of those.

It has several files, that contain the following information

 * load_embedding.py - attempts to provide you with an explanation of loading embedding... but the current approach 
 is not working properly
 * word2vec.py - class implementation. Too few comments in the code and no real explanation yet. Also the classes 
 have not been tested yet.
 * example_word2vec.py - a start has been made. Not finished yet. 
 * example_gensim.py - example usage with word2vec defined in gensim
    
 
 ### Result from example_gensim.py
 
 Just to show you some preliminary results of a very basic run... in the example_gensim.py script we load imbd movie 
 reviews, put those reviews in a word2vec model, and then plot the word-vectors with tsne on a 2D plane.
 
  
 The movie reviews contain positive and negative comments.. here the red dots are positive, and the purple dots are 
 negative comments
 
![Voila](../plot/wordembedding/imbd_movie_reviews.png "A simple clustering of movie reviews")

As you can see, the different type of comments are not fully separated. One interesting thing to try out next, is to 
see which words are mapped where. Because maybe movie names are being grouped together, instead of the sentiment of 
the comment.

Hence, more research is needed@

    
    