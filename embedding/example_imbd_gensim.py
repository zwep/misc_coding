# encoding: utf-8

"""
In this file we show a way to encode words through word2vec.

We give an example where we gather some data from the imbd-movie set for some comments on movies... and see if we can
deduce whether we can split the negative and positve comments.
For this we use the package gensim in python (of course).

"""

import gensim.models
import nltk
import random
import numpy as np

# These helper functions are created by myself.. you can edit them of course!
from helper.plotfunction import tsne_plot
from helper.miscfunction import transform_to_color, read_txt_file

# Download imbd movie reviews at
# http://ai.stanford.edu/~amaas/data/sentiment/
location_imbd = r'D:\data\text\imdb\train'

# Here we define the function that can load the imbd data based on the intent...
# Here we actually load the positive and negative reviews
n_articles = None  # Taking all (12500 comments), will require some patience
n_articles = 100  # Taking 100 comments is very fast...

pos_review = read_txt_file(location_imbd + r'\pos', n_files=n_articles)  # Simple example of how the data is loaded.
neg_review = read_txt_file(location_imbd + r'\neg', n_files=n_articles)  # Simple example of how the data is loaded.
if n_articles is None:
    n_articles = len(pos_review)

# Combine everything into one big set
input_data = pos_review + neg_review
target_data = [1] * n_articles + [0] * n_articles

# Tokenize the sentences to have them as a proper input for the Word2Vec model
input_data_token = [nltk.tokenize.word_tokenize(x) for x in input_data]

# The parameters here are best to be looked up by help (gensim.models.Word2Vec)
# The choice of parameters is rather arbitrary, nothing is optimized
model = gensim.models.Word2Vec(input_data_token, min_count=10, iter=20, sg=1, size=300, alpha=0.2)

# If you want to visualize them.. you can plot it like this.
# By using the target data, we convert the different classes to a color.. and then plot that
# You can also edit the tsne_plot() function to plot labels per data point.. but this gets messy very fast.
vocab = list(model.wv.vocab)
X = model[vocab]

# In order to represent a comment into one single vector, we need to concat the vectors of the individual words in
# some way...
# This takes a lot of time.. maybe the model-object itself already has some... prepocessed stuff (maybe)
input_data_token_index = [[vocab.index(x) for x in y if x in vocab] for y in input_data_token]
input_data_token_sum = [(i, sum(X[y])) if len(y) > 0 else (i, -1) for i, y in enumerate(input_data_token_index)]

# Clean out the ones that are 'broken'
index_data_token_sum, input_data_token_sum2 = zip(*[x for x in input_data_token_sum if not isinstance(x[1], int)])
input_data_token_sum_stacked = np.stack(input_data_token_sum2)
# Also filter out the ones in the target...
target_data_filter = np.array(target_data)[list(index_data_token_sum)]

# Now sample (stratified.. in a way...) from the target distribution
# This is because visualizing them via tsne is quite intensive.. os if you want to quickly check your results,
# it is best to sub sample them
sampled_data = []
n_pos = 0
n_neg = 0
n_max = 500
# But also shuffle it.. so we dont get the top-X ones every time
target_data_shuffle = list(enumerate(target_data_filter))
random.shuffle(target_data_shuffle)

# Add them to a list that contains n_max positive and n_max negative comments
for i, x in target_data_shuffle:
    if x == 1 and n_pos < n_max:
        n_pos += 1
        sampled_data.append((i, x))
    elif x == 0 and n_neg < n_max:
        n_neg += 1
        sampled_data.append((i, x))

# Strip both the index and the value.. with this we can easily index the feature values
index_target, value_target = zip(*sampled_data)
# Transform the target to a color..
target_data_color = transform_to_color(value_target)
# And plot them.
tsne_plot(input_data_token_sum_stacked[list(index_target)], target_data_color)
