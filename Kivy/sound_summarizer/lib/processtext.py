"""
libraries
"""
import nltk
from nltk.stem.snowball import DutchStemmer
from nltk.tokenize import sent_tokenize,word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

import collections
import string

import numpy as np

import matplotlib.pyplot as plt
"""
 classes
"""
		
class CleanText:
	"""
	This class can clean your string by stemming or removing stopwords.
	Most NLTK things are in English.. and I wanted to create my own. So here it is.
	"""
	def __init__(self,text,language = 'dutch'):
		self.text = text
		self._stopwords = set(nltk.corpus.stopwords.words(language) + list(string.punctuation))
		self._trans_punc = str.maketrans(string.punctuation, ' '*len(string.punctuation))
		
		if language == 'dutch':
			self.stemmer = DutchStemmer()
		elif language == 'english':
			self.stemmer = PorterStemmer()
		
	def _remove_stopwords_string(self,string):
		"""
		Removes stopwords from a splitted string, and then joining it agian.
		"""
		word = string.lower().split()
		string_filtered = [w for w in word if not w in self._stopwords]
		return ' '.join(string_filtered)
		
	def _stem_string(self,string):
		"""
		Stems the given string by splitting it, and then joining it again.
		"""
		word = string.lower().split()
		string_stemmed = [self.stemmer.stem(w) for w in word]
		return ' '.join(string_stemmed)
		
	def stem_text(self):
		"""
		Stems the given articles 
		"""	 
		article_stemmed = [self._stem_string(a) for a in self.text]
		return CleanText(article_stemmed)
	
	def remove_stopwords_text(self):
		"""
		Stems the given articles 
		"""	 
		article_filtered = [self._remove_stopwords_string(a) for a in self.text]
		return CleanText(article_filtered)
		
	def remove_punc_text(self):
		"""
		Removes punctuation of the content of a list
		"""	
		article_no_punc = [x.translate(self._trans_punc).lower() for x in self.text]	
		return CleanText(article_no_punc)

		
# PLEASEPLEASEPLEASE
# CHECK/TEST if the dictionary is keeping good reference to the actualy places...
# Because dictionaries can order their keys sometimes allphabetically, and sometimes not.
# Sometimes this is desired and sometimes not...		
class TFIDF():
	"""
	Making TFIDF possible via my own class. Reason being, the TfidfVectorizer from sklearn
	was (IMO) too complex and did not (clearly) offer any variation on the tf and idf.
	
	No preprocessing is done here, you can do this yourself. That is so much clearer (IMO).
	"""
	def __init__(self,document_list):
		self.document_list = document_list
		self.vocab_document = np.unique(np.concatenate([x.split() for x  in self.document_list],axis = 0))
		self.vocab_dict = dict(enumerate(self.vocab_document))
		self.vocab_dict = {v: k for k,v in self.vocab_dict.items()}
		self.word_N = len(self.vocab_dict)
		self.idf_N = len(self.document_list)
		self._TF()
		self._IDF()
	"""
	Initializes the Term Frequency for the given documents.
	All variations are dependent on this one, as well as the IDF.
	Output is the (#documents x # words in vocab)
	"""
	def _TF(self):
		self.TF_count = np.ndarray(shape = (self.idf_N,self.word_N))	
		# 		
		for i_index, x in enumerate(self.document_list):
			words = np.sort(x.split()) 
			words_unique = np.unique(words)
			words_count = collections.Counter(words)
			words_count = collections.OrderedDict(sorted(words_count.items()))

			position = list(map(self.vocab_dict.get,words_unique))			
			value_ftd = np.array(list(words_count.values()))
			
			self.TF_count[i_index,position]  = value_ftd		 
		
		return self.TF_count
		
	"""
	The binary version of the Term Frequency
	"""	
	def TF_bin(self): 
		TF_bin = (self.TF_count != 0).astype(int)
		return TF_bin 
		
	"""
	The scaled version of the Term Frequency
	"""	
	def TF_freq(self): 
		ftd_len = self.TF_count.sum(axis = 1)
		TF_freq = self.TF_count / ftd_len[:,None]
		return TF_freq 
		
	"""
	The log version of the Term Frequency
	"""			
	def TF_log_freq(self): 
		TF_log_freq = 1 + np.log(np.ma.log(self.TF_count)).filled(0)
		return TF_log_freq 
	
	"""
	The augmented freuqency version of the Term Frequency
	"""	
	def TF_augm_freq(self): 
		ftd_max = self.TF_count.max(axis = 1)
		TF_augm_freq = 0.5 + 0.5 * self.TF_count / ftd_max[:,None]
		return TF_augm_freq 
	
	"""
	The Inverse Document Frequency. All IDFs outputs a vector the size of the vocab.
	"""
	def _IDF(self):  
		idf_count = collections.Counter(np.nonzero(self.TF_count)[1])
		idf_count = collections.OrderedDict(sorted(idf_count.items()))
		self.IDF_weight = np.array(list(idf_count.values()))
		self.IDF = np.log(self.idf_N/(1+self.IDF_weight))
		return self.IDF
	
	"""
	The smoothed version of the Inverse Document Frequency
	"""
	def IDF_smooth(self): 
		idf_smooth = np.log(1 + self.idf_N/self.IDF_weight)
		return idf_smooth 
	
	"""
	The max version of the Inverse Document Frequency
	"""
	def IDF_max(self): 
		idf_max = np.log(max(self.IDF_weight)/(1+self.IDF_weight))
		return idf_max	 
		
	"""
	The probabilistic version of the Inverse Document Frequency
	"""	
	def IDF_prob(self): 
		idf_prob = np.log((self.idf_N - self.IDF_weight)/self.IDF_weight)
		return idf_prob 
	

"""
functions
"""



# use the standard tfidf thing of sklearn					
def tfidf_doc(article):
	tfidf = TfidfVectorizer()
	tf = tfidf.fit_transform(article)
	tf_name= tfidf.get_feature_names()
	tf_document = tf.todense()
	
	return tf_name,tf_document
	
# plotting some data with tsne	
def tsne_plot(data,color):
	# Caluclate TSNE clustering
	tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=1000, method='exact')
	tsne_document = tsne.fit_transform(data)

	plt.figure(figsize=(9, 9))  # in inches
	for i_index, i_color in enumerate(color):
		x, y = tsne_document[i_index, :]
		plt.scatter(x, y,color = i_color)				 
	
	plt.show()
	
# Simple function for plotting and scoring a simple linear regression.
def score_logistic_regression(features,target,feature_label,target_label = None,plot = False,p_train = 0.7):
	if target_label is None:
		target_label = target
		
	N = len(features)
	N_train = int(np.round(N*p_train))
	index_full = range(N)
	index_train = np.random.choice(index_full,N_train,replace = False)
	index_test = list(set.difference(set(index_full),set(index_train)))	

	if plot == True:
		tsne_plot(features,target_label)

	model = sklearn.linear_model.logistic.LogisticRegression()
	linear_reg = model.fit(features[index_train],target[index_train])
	test_score = linear_reg.score(features[index_test],target[index_test])
	train_score = linear_reg.score(features[index_train],target[index_train])

	betahat = linear_reg.coef_[0,:]
	betahat = np.abs(betahat)

	N_max = 10
	print(np.array(feature_label)[betahat.argsort()[-N_max:][::-1]])

	return test_score, train_score
