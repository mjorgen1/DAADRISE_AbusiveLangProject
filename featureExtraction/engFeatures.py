import pandas as pd
import os
import copy
from sklearn.feature_extraction.text import CountVectorizer

# Check the location of current working directory and move the dataset to that directory
os.getcwd()
# Or change the working directory
os.chdir('C:\\Users\\mikec\\Documents')

# Raw data
clean_data = pd.read_csv("EnglishCleanedTrainingData.csv").fillna(' ')
tweets = copy.deepcopy(clean_data['tweet'])

# you can set the n-gram range to 1,2 to get unigrams as well as bigrams
bv = CountVectorizer(ngram_range=(2, 2))
bv_matrix = bv.fit_transform(tweets)

bv_matrix = bv_matrix.toarray()
vocab = bv.get_feature_names()
pd.DataFrame(bv_matrix, columns=vocab)


