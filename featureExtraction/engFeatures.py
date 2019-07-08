import pandas as pd
import os
import copy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Check the location of current working directory and move the dataset to that directory
os.getcwd()
# Or change the working directory
os.chdir('C:\\Users\\mikec\\Documents')

# Raw data
clean_data = pd.read_csv("EnglishCleanedTrainingData.csv").fillna(' ')
tweets = copy.deepcopy(clean_data['tweet'])


# Returns bag of n-grams
def bag_of_n_grams(text):
    bv = CountVectorizer(ngram_range=(1, 2))
    bv_matrix = bv.fit_transform(text)
    bv_matrix = bv_matrix.toarray()
    bv_vocab = bv.get_feature_names()
    bv_data = pd.DataFrame(bv_matrix, columns=bv_vocab)
    return bv_data


def tf_idf(text):
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    tv_matrix = tv.fit_transform(text)
    tv_matrix = tv_matrix.toarray()
    tv_vocab = tv.get_feature_names()
    tv_data = pd.DataFrame(np.round(tv_matrix, 2), columns=tv_vocab)
    return tv_data
