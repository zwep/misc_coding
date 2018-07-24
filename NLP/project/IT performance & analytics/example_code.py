import collections  # Used to handle collections...
import string  # Used to handle strings more easy
import re  # Using this to execute regular expressions
import nltk  # Using this for NLP-like functionanilties
import pandas as pd  # Using to read .csv files in as tables
import os  # Used to navigate the operating system

dir_data = r'C:\Users\C35612.LAUNCHER\1_Data_Innovation_Analytics\project\NLP\Business Case Description\IT performance & analytics'

"""
Loading data
"""
os.chdir(dir_data)  # Change the directory to the location of the data
# Here we read in the data. After some trying we needed to specify a different encoding (latin), the standard encoding is utf-8
# Also we need to specify the separator of the .csv file, otherwise we had some column issue
# Because there were some columns with mixed type, we will just tell that the dtype of all columns
survey_data = pd.read_csv('IT_Office_Surveys_CMM_H2_data.csv', encoding='latin', sep=';', dtype=str)  
print(survey_data.columns)  # See which columns are there..

survey_data.Remark  # With this we can select a column
survey_data['Remark']  # Or with this

"""
Cleaning data
"""
# Before we can start counting words.. we need to clean some data

clean_survey_data = survey_data[~survey_data['Remark'].isnull()]  # With this we exclude all the NaN valued rows in Remark

# We can also clean some other rows that contain n/a values
# We can find these values as followed
v = clean_survey_data.Remark.values  # First take the values of the column
[x for x in v if re.search('n/a',x)]  # Then search with regular expression of the string "n/a" is present in the element
# You will see that we won't get the result we expected... this is an important part of data analysis.
# So we will just leave the n/a's for now.

# We need to keep in mind that we have some punctuations and stopwords in our text
# Therefore we create a set of these words that can be used later to filter out certain tokens.
english_stopwords = set(nltk.corpus.stopwords.words(fileids='english') + list(string.punctuation))
print(english_stopwords)  # Now you can see them for yourself.

"""
Counting data
"""

# Now let's count some words!

# We already have all the content stored in a list called v
# So let's tokenize the senteces in the list to words 
v_words = [nltk.tokenize.word_tokenize(x) for x in v]  # This procedure is called list comprehension, you should look it up.
# Here we remove the english stopwords per sentence, but we maintain the sentence structure
v_words = [[x.lower() for x in b if x not in english_stopwords] for b in v_words]

# To count the words, we like to have everything in one list
v_list = [x for b in v_words for x in b]  # This is a double-list comprehension
v_word_count = collections.Counter(v_list)  # Here we count the words in the list. The result is a generator.. which can be made 'explicit' by calling list(v_word_count)
v_word_count.most_common(10)
 
# But maybe we want to count how popular some consectuive words are..
# We can do that by counting bi-grams per review comment
# (To try tri-grams, or four-grams etc.. just change n_gram to 3, 4, ...)
# (It is a nice and simple exercise to create a function for this.)
n_gram = 2
v_words_n_gram = [list(nltk.ngrams(x,n_gram)) for x in v_words]  # Here we create the n-grams
v_words_n_gram_flat = [x for b in v_words_n_gram for x in b]  # Now we flatten them, so that we can count them more easy
v_n_gram_count = collections.Counter(v_words_n_gram_flat)  # And the counting operation again
print(v_n_gram_count.most_common(10))  # Looking at the 10 most popular ones


"""
Counting data per group
"""

# We have counted some nice properties... but not by any group in which we might be interested.
# So let's say we want to know the most popular words per vendor... then we can do the following

# We are going to put the above sequence of instructions into a function
# And then apply this function to a grouped-by dataframe (from the package pandas)

def count_dataframe_column(x):
	n_gram = 2  # This can be of value 1, 2, 3, ...
	y = [nltk.tokenize.word_tokenize(x) for x in list(x)]
	y_clean = [[x.lower() for x in b if x not in english_stopwords] for b in y]	
	y_ngram = [list(nltk.ngrams(x,n_gram)) for x in y_clean] 
	y_ngram_flat = [x for b in y_ngram for x in b] 
	y_ngram_count = collections.Counter(y_ngram_flat)  
	
	return y_ngram_count.most_common(10)

clean_survey_data.groupby('Vendor').Remark.apply(count_dataframe)  # Here we get the word-counts per vendor
clean_survey_data.groupby('Unit').Remark.apply(count_dataframe)  # Here we get the word-counts per Unit
clean_survey_data.groupby(['Domein','Subdomein']).Remark.apply(count_dataframe) # Here we get the word-counts per Domein and Subdomein combination
