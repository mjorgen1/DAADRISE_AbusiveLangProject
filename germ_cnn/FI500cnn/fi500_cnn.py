import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import nltk
from nltk.stem.cistem import Cistem
import string
import re
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from textstat.textstat import *
import matplotlib.pyplot as plt
from imblearn.under_sampling import (CondensedNearestNeighbour)
import seaborn
from textblob_de import TextBlobDE as TextBlob
import warnings
import random
import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

# Raw data
train_data = pd.read_csv("/home/riseadmin/DAADRISE_AbusiveLangProject/tpot/GermanTrainingData.txt", sep='\t', names=['tweet', 'coarse', 'labels'])
test_data = pd.read_csv("/home/riseadmin/DAADRISE_AbusiveLangProject/tpot/GermanTestingData.txt", sep='\t', names=['tweet', 'coarse', 'labels'])
df = pd.concat([train_data, test_data], ignore_index=True)
del train_data
del test_data

tweets=df.tweet

# function to change accents for text processing purposes
def convert_umlauts(text):
    temp = text
    temp = temp.replace('ä', 'ae')
    temp = temp.replace('ö', 'oe')
    temp = temp.replace('ü', 'ue')
    temp = temp.replace('Ä', 'Ae')
    temp = temp.replace('Ö', 'Oe')
    temp = temp.replace('Ü', 'Ue')
    temp = temp.replace('ß', 'ss')
    return temp

# call the tweets below to change accented word to their roots
for i in range(0, len(tweets)):
    curr = tweets.iloc[i]
    tweets.iloc[i] = convert_umlauts(curr)

stopwords=stopwords = nltk.corpus.stopwords.words("german")

other_exclusions = ["lbr", "|lbr|", "»"]
stopwords.extend(other_exclusions)

stemmer = Cistem()

def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-ßA-ß]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    return parsed_text

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-ßA-ß]+", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-ßA-ß.,!?]+", tweet.lower())).strip()
    return tweet.split()

vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    preprocessor=preprocess,
    ngram_range=(1, 3),
    stop_words=stopwords,
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=3000,
    min_df=5,
    max_df=0.75
    )

warnings.simplefilter(action='ignore', category=FutureWarning)

#Construct tfidf matrix and get relevant scores
tfidf = vectorizer.fit_transform(tweets).toarray()
vocab = {v:i for i, v in enumerate(vectorizer.get_feature_names())}
idf_vals = vectorizer.idf_
idf_dict = {i:idf_vals[i] for i in vocab.values()} #keys are indices; values are IDF scores


# Now get other features

def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.

    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-ßA-ß]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return (parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'), parsed_text.count('HASHTAGHERE'))


def other_features(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    sentiment = TextBlob("tweet").sentiment

    words = preprocess(tweet)  # Get text only

    syllables = textstat.syllable_count(words, lang='de_DE')
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables + 0.001)) / float(num_words + 0.001), 4)
    num_unique_terms = len(set(words.split()))

    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words) / 1.0) + float(11.8 * avg_syl) - 15.59, 1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015 * (float(num_words) / 1.0) - (84.6 * float(avg_syl)), 2)

    twitter_objs = count_twitter_objs(tweet)
    features = [FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment[0], twitter_objs[2], twitter_objs[1], twitter_objs[0]]
    # features = pandas.DataFrame(features)
    return features


def get_feature_array(tweets):
    feats = []
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)


# Changing string labels to numeric labels
def string_to_numeric(x):
    if x == 'OTHER' or x == 'PROFANITY':
        return 0
    if x == 'INSULT':
        return 1
    if x == 'ABUSE':
        return 2

other_features_names = ["FKRA", "FRE","num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total", \
                        "num_terms", "num_words", "num_unique_words", "sentiment", "num_hashtags", "num_mentions", "num_urls"]

feats = get_feature_array(tweets)

M = np.concatenate([tfidf,feats],axis=1)

M.shape

#Finally get a list of variable names
variables = ['']*len(vocab)
for k,v in vocab.items():
    variables[v] = k

feature_names = variables+other_features_names

X = pd.DataFrame(M)
y = df['labels'].apply(string_to_numeric)
X.columns = feature_names

# Feature Selection
print('starting feature selection: FI 500')
model = ExtraTreesClassifier(n_estimators=10) #n_estimators=300
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
top_feat_impt = feat_importances.nlargest(500)
print(top_feat_impt) # prints out the n best features

# Saving the top n features to a dataframe
list_names = top_feat_impt.axes
best_features = pd.DataFrame()
#print(X.columns.get_loc(list[0][0])) # how to get the index of the column/name from the feature selected names
for i in range(0, 500):
    curr_column_name = list_names[0][i]
    curr_column_index = X.columns.get_loc(curr_column_name)
    curr_column_vals = X.iloc[:, curr_column_index]
    best_features[curr_column_name] = curr_column_vals

X = pd.DataFrame(best_features)
feature_names = best_features.columns
X.columns = feature_names
best_features = top_feat_impt
best_features.to_csv('top_FI_features500.csv')
print('FI done!')

print(len(X))
print(len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

y_train = y_train.to_frame(name='labels')
y_test = y_test.to_frame(name='labels')

X_test.to_csv('X_test.csv', index=None, header=True, encoding='utf-8')
y_test.to_csv('y_test.csv', index=None, header=True, encoding='utf-8')

# Condensed Nearest Neighbor
from imblearn.under_sampling import CondensedNearestNeighbour
print("Condensed Nearest Neighbour!!")
cnn = CondensedNearestNeighbour(random_state=2)
X_res, y_res = cnn.fit_resample(X_train, y_train)
X_train = pd.DataFrame(X_res)
X_train.columns = feature_names
y_train = pd.DataFrame()
y_train['labels'] = y_res
X_train.to_csv('X_cnn.csv', index=None, header=True, encoding='utf-8')
y_train.to_csv('y_cnn.csv', index=None, header=True, encoding='utf-8')
print("Done!")