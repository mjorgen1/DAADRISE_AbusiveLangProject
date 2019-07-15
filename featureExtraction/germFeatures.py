import pandas as pd
import os
import copy
import numpy as np
import re
import nltk
import textstat
import emoji
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Check the location of current working directory and move the dataset to that directory
os.getcwd()
# Or change the working directory
os.chdir('C:\\Users\\mikec\\Documents')

# Raw data
clean_data = pd.read_csv("GermanCleanedTrainingData.csv")
clean_tweets = copy.deepcopy(clean_data['cleaned_tweet'])
tweets = copy.deepcopy(clean_data['tweet'])

''' Linguistic Feature Extraction '''


def text_length(text):
    return len(text)


def number_of_tokens(text):
    tokens = nltk.word_tokenize(text)
    return len(tokens)


def number_of_mentions(text):
    return len(re.findall(r"@\S+", text))


def number_of_hashtags(text):
    return len(re.findall(r"#\S+", text))


def number_of_links(text):
    return len(re.findall(r"http\S+", text))


# Input: a text, Output: how many emoticons
def number_of_emoticons(text):
    count = 0
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            count += 1
    return count


''' N-gram Features '''


# Returns bag of n-grams
def bag_of_n_grams(text, min_n, max_n):
    bv = CountVectorizer(ngram_range=(min_n, max_n), max_features=1000)
    bv_matrix = bv.fit_transform(text).toarray()
    bv_vocab = bv.get_feature_names()
    bv_data = pd.DataFrame(bv_matrix, columns=bv_vocab)
    return bv_data


def tf_idf(text, min_n, max_n):
    tv = TfidfVectorizer(ngram_range=(min_n, max_n), max_features=1000)
    tv_matrix = tv.fit_transform(text).toarray()
    tv_vocab = tv.get_feature_names()
    tv_data = pd.DataFrame(np.round(tv_matrix, 2), columns=tv_vocab)
    return tv_data


''' Fasttext'''

''' Testing '''


# Remove any user names referred in the text
def remove_user_names(text):
    text = re.sub(r'@\S+', '', text)
    return text


# Remove hashtags
def remove_hashtags(text):
    text = re.sub(r'#\S+', '', text)
    return text


# Remove links in the text
def remove_links(text):
    text = re.sub(r'http\S+', '', text)
    return text


# Remove underscore in the text
def remove_underscore(text):
    text = text.replace('_', '')
    return text


def remove_emojis(text):
    return ''.join(c for c in text if c in emoji.UNICODE_EMOJI)


tl, nt, nom, noh, nol, noem = [], [], [], [], [], []
fre, fkg, fs, si, ari, cli, lwf, dcrs = [], [], [], [], [], [], [], []


for i in range(0, len(tweets)):
    # '''
    tl.append(text_length(tweets[i]))
    nom.append(number_of_mentions(tweets[i]))
    noh.append(number_of_hashtags(tweets[i]))
    nol.append(number_of_links(tweets[i]))
    noem.append(number_of_emoticons(tweets[i]))
    nt.append(number_of_tokens(tweets[i]))

    tweets[i] = remove_user_names(tweets[i])
    tweets[i] = remove_hashtags(tweets[i])
    tweets[i] = remove_links(tweets[i])
    tweets[i] = remove_underscore(tweets[i])
    tweets[i] = remove_emojis(tweets[i])

    fre.append(textstat.flesch_reading_ease(tweets[i]))
    fkg.append(textstat.flesch_kincaid_grade(tweets[i]))
    fs.append(textstat.gunning_fog(tweets[i]))
    si.append(textstat.smog_index(tweets[i]))
    ari.append(textstat.automated_readability_index(tweets[i]))
    cli.append(textstat.coleman_liau_index(tweets[i]))
    lwf.append(textstat.linsear_write_formula(tweets[i]))
    dcrs.append(textstat.dale_chall_readability_score(tweets[i]))
    # '''

features = pd.DataFrame()
# '''
features['text length'] = tl
features['number of words'] = nt
features['number of mentions'] = nom
features['number of hashtags'] = noh
features['number of links'] = nol
features['number of emoticons'] = noem
features['flesch reading ease'] = fre
features['flesch kincaid grade'] = fkg
features['gunning fog'] = fs
features['smog index'] = si
features['automated readability index'] = ari
features['coleman liau index'] = cli
features['linsear write formula'] = lwf
features['dale chall readability score'] = dcrs

bn_data = bag_of_n_grams(clean_tweets, 1, 3)
tf_data = tf_idf(clean_tweets, 1, 3)
# '''

