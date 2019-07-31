# here we will have the code to run on the server
import pandas as pd
import numpy as np
from tpot import TPOTClassifier
import pickle # issue downloading to server
import sys # issue downloading to server
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from nltk.stem.cistem import Cistem
import string # issue downloading to server
import re # issue downloading to server
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn
from textblob_de import TextBlobDE as TextBlob
import warnings # issue downloading to server

# Raw data
train_data = pd.read_csv("GermanTrainingData.txt", sep='\t', names=['tweet', 'coarse', 'labels'])
test_data = pd.read_csv("GermanTestingData.txt", sep='\t', names=['tweet', 'coarse', 'labels'])
df = pd.concat([train_data, test_data], ignore_index=True)
del train_data
del test_data

tweets=df.tweet

stopwords=stopwords = nltk.corpus.stopwords.words("german")

other_exclusions = ["lbr"]
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

#Get POS tags for tweets and save as a string
tweet_tags = []
for t in tweets:
    tokens = basic_tokenize(preprocess(t))
    tags = nltk.pos_tag(tokens)
    tag_list = [x[1] for x in tags]
    tag_str = " ".join(tag_list)
    tweet_tags.append(tag_str)

#We can use the TFIDF vectorizer to get a token matrix for the POS tags
pos_vectorizer = TfidfVectorizer(
    tokenizer=None,
    lowercase=False,
    preprocessor=None,
    ngram_range=(1, 3),
    stop_words=None,
    use_idf=False,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=2000,
    min_df=5,
    max_df=0.75,
    )

#Construct POS TF matrix and get vocab dict
pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
pos_vocab = {v:i for i, v in enumerate(pos_vectorizer.get_feature_names())}

#Now get other features
sentiment_analyzer = VS()

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
    return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))

def other_features(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    sentiment = TextBlob("tweet").sentiment

    words = preprocess(tweet) #Get text only

    syllables = textstat.syllable_count(words, lang='de_DE')
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
    num_unique_terms = len(set(words.split()))

    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)

    twitter_objs = count_twitter_objs(tweet)
    features = [FKRA, FRE,syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment[0],twitter_objs[2], twitter_objs[1], twitter_objs[0]]
    #features = pandas.DataFrame(features)
    return features

def get_feature_array(tweets):
    feats=[]
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

#Now join them all up
M = np.concatenate([tfidf,pos,feats],axis=1)

M.shape

#Finally get a list of variable names
variables = ['']*len(vocab)
for k,v in vocab.items():
    variables[v] = k

pos_variables = ['']*len(pos_vocab)
for k,v in pos_vocab.items():
    pos_variables[v] = k

feature_names = variables+pos_variables+other_features_names

X = pd.DataFrame(M)
y = df['labels'].apply(string_to_numeric)
X.columns = feature_names

'''
# Univariate feature selection from sklearn
bestfeatures = SelectKBest(score_func=f_classif, k=3000)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

# concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print('Univariate Selection features found, use getUnivariateData() to get the features')
# Extract the top n features
uni_selected_feat = featureScores.nlargest(3000,'Score')
print(uni_selected_feat) # print out the top n features selected

# Saving the top n features to a data frame
#print(a.iloc[0].name) # how to get the column # for the ith feature
#print(a.iloc[0][0]) # how to get the header column
top_univariate_features = pd.DataFrame()
for i in range(0, 3000):
    curr_column_vals = X.iloc[:, uni_selected_feat.iloc[i].name]
    curr_column_name = uni_selected_feat.iloc[i][0]
    top_univariate_features[curr_column_name] = curr_column_vals

X = pd.DataFrame(top_univariate_features)
X.columns = top_univariate_features.columns
print(X.columns)
feature_names = top_univariate_features.columns
'''

print('x len:')
print(len(X))
print('y len:')
print(len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

'''
# Try undersampling from imblearn
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)
X_train = pd.DataFrame(X_res)
X_train.columns = feature_names
y_train = pd.DataFrame(y_res)
y_train.columns = ['labels']
'''

tpot = TPOTClassifier(generations=3, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
