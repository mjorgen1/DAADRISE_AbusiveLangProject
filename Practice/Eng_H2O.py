import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn

df = pd.read_csv("labeled_data.csv")

tweets=df.tweet

stopwords = nltk.corpus.stopwords.words("english")

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

stemmer = PorterStemmer()

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
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    return parsed_text

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]+", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]+", tweet.lower())).strip()
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
    max_features=10000,
    min_df=5,
    max_df=0.75
    )

import warnings
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
    max_features=5000,
    min_df=5,
    max_df=0.75,
    )

#Construct POS TF matrix and get vocab dict
pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
pos_vocab = {v:i for i, v in enumerate(pos_vectorizer.get_feature_names())}

# Now get other features
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
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
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
    sentiment = sentiment_analyzer.polarity_scores(tweet)

    words = preprocess(tweet)  # Get text only

    syllables = textstat.syllable_count(words)
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
    retweet = 0
    if "rt" in words:
        retweet = 1
    features = [FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], retweet]
    # features = pandas.DataFrame(features)
    return features


def get_feature_array(tweets):
    feats = []
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)

other_features_names = ["FKRA", "FRE","num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total", \
                        "num_terms", "num_words", "num_unique_words", "vader neg","vader pos","vader neu", \
                        "vader compound", "num_hashtags", "num_mentions", "num_urls", "is_retweet"]

feats = get_feature_array(tweets)

#Now join them all up
M = np.concatenate([tfidf , pos, feats], axis=1)

print(M.shape)

#Finally get a list of variable names
variables = ['']*len(vocab)
for k,v in vocab.items():
    variables[v] = k

pos_variables = ['']*len(pos_vocab)
for k,v in pos_vocab.items():
    pos_variables[v] = k

feature_names = variables+pos_variables+other_features_names

X = pd.DataFrame(M)
y = df['class'].astype(int)
X.columns = feature_names

#Feature selection
from sklearn.feature_selection import SelectKBest, f_classif
# Univariate Selection -- apply SelectKBest class to extract top n best features
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
top_univariate_features = pd.DataFrame()
for i in range(0, 3000):
    curr_column_vals = X.iloc[:, uni_selected_feat.iloc[i].name]
    curr_column_name = uni_selected_feat.iloc[i][0]
    top_univariate_features[curr_column_name] = curr_column_vals
X = top_univariate_features

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

X_test.to_csv('X_test.csv', index=None, header=True, encoding='utf-8')
y_test = y_test.to_frame(name='labels')
y_test.to_csv('y_test.csv', index=None, header=True, encoding='utf-8')

from imblearn.under_sampling import RandomUnderSampler
print("Random Under Sampler!!")
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)
X_rus = pd.DataFrame(X_res)
X_rus.columns = uni_selected_feat['Specs']
y_rus = pd.DataFrame()
y_rus['labels'] = y_res
X_rus.to_csv('X_rus.csv', index=None, header=True, encoding='utf-8')
y_rus.to_csv('y_rus.csv', index=None, header=True, encoding='utf-8')
print("Done!")

from imblearn.under_sampling import ClusterCentroids
print("Cluster Centroids!!")
cc = ClusterCentroids(random_state=2)
X_res, y_res = cc.fit_resample(X_train, y_train)
X_cc = pd.DataFrame(X_res)
X_cc.columns = uni_selected_feat['Specs']
y_cc = pd.DataFrame()
y_cc['labels'] = y_res
X_cc.to_csv('X_cc.csv', index=None, header=True, encoding='utf-8')
y_cc.to_csv('y_cc.csv', index=None, header=True, encoding='utf-8')
print("Done!")

from imblearn.under_sampling import CondensedNearestNeighbour
print("Condensed Nearest Neighbour!!")
cnn = CondensedNearestNeighbour(random_state=2)
X_res, y_res = cnn.fit_resample(X_train, y_train)
X_cnn = pd.DataFrame(X_res)
X_cnn.columns = uni_selected_feat['Specs']
y_cnn = pd.DataFrame()
y_cnn['labels'] = y_res
X_cnn.to_csv('X_cnn.csv', index=None, header=True, encoding='utf-8')
y_cnn.to_csv('y_cnn.csv', index=None, header=True, encoding='utf-8')
print("Done!")

from imblearn.under_sampling import EditedNearestNeighbours
print("Edited Nearest Neighbours!!")
enn = EditedNearestNeighbours()
X_res, y_res = enn.fit_resample(X_train, y_train)
X_enn = pd.DataFrame(X_res)
X_enn.columns = uni_selected_feat['Specs']
y_enn = pd.DataFrame()
y_enn['labels'] = y_res
X_enn.to_csv('X_enn.csv', index=None, header=True, encoding='utf-8')
y_enn.to_csv('y_enn.csv', index=None, header=True, encoding='utf-8')
print("Done!")

from imblearn.under_sampling import AllKNN
print("AllKNN!!")
allknn = AllKNN()
X_res, y_res = allknn.fit_resample(X_train, y_train)
X_allknn = pd.DataFrame(X_res)
X_allknn.columns = uni_selected_feat['Specs']
y_allknn = pd.DataFrame()
y_allknn['labels'] = y_res
X_allknn.to_csv('X_allknn.csv', index=None, header=True, encoding='utf-8')
y_allknn.to_csv('y_allknn.csv', index=None, header=True, encoding='utf-8')
print("Done!")

from imblearn.under_sampling import NeighbourhoodCleaningRule
print("Neighourhood Cleaning Rule!!")
ncr = NeighbourhoodCleaningRule()
X_res, y_res = ncr.fit_resample(X_train, y_train)
X_ncr = pd.DataFrame(X_res)
X_ncr.columns = uni_selected_feat['Specs']
y_ncr = pd.DataFrame()
y_ncr['labels'] = y_res
X_ncr.to_csv('X_ncr.csv', index=None, header=True, encoding='utf-8')
y_ncr.to_csv('y_ncr.csv', index=None, header=True, encoding='utf-8')
print("Done!")

from imblearn.under_sampling import OneSidedSelection
print("One Sided Selection!!")
oss = OneSidedSelection(random_state=2)
X_res, y_res = oss.fit_resample(X_train, y_train)
X_oss = pd.DataFrame(X_res)
X_oss.columns = uni_selected_feat['Specs']
y_oss = pd.DataFrame()
y_oss['labels'] = y_res
X_oss.to_csv('X_oss.csv', index=None, header=True, encoding='utf-8')
y_oss.to_csv('y_oss.csv', index=None, header=True, encoding='utf-8')
print("Done!")






