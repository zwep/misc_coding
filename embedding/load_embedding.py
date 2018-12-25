# encoding: utf-8

"""
Here we show how to load the Google News model (for word2vec) into python using gensim.
(check for some more info:  http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/)

After that, we also show how to load the Dutch embedding.. (https://github.com/clips/dutchembeddings)
After that we show how to retrain it or infer statistics out of it.
"""

import gensim


loc_data = r'D:\data\text\Wordembed'
model = gensim.models.KeyedVectors.load_word2vec_format(loc_data + '\GoogleNews-vectors-negative300.bin.gz',
                                                        binary=True)
katvec = model['cat']
print(katvec.shape)
print(model.most_similar('cat'))


model = gensim.models.KeyedVectors.load_word2vec_format(loc_data + 'dutch_wikipedia_320.txt', binary=False)
dogvec = model['hond']
print(dogvec.shape)
print(model.most_similar('hond'))
