import pandas as pd
import os
import copy
import numpy as np
import re
import nltk
import readability
from functools import partial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Check the location of current working directory and move the dataset to that directory
os.getcwd()
# Or change the working directory
os.chdir('C:\\Users\\mikec\\Documents')

# Raw data
clean_data = pd.read_csv("EnglishCleanedTrainingData.csv")
clean_tweets = copy.deepcopy(clean_data['cleaned_tweet'])
tweets = copy.deepcopy(clean_data['tweet'])

''' Linguistic Feature Extraction '''
def text_length(text):
    return len(text)


def number_of_tokens(text):
    tokens = nltk.word_tokenize(text)
    return len(tokens)


def is_retweet(text):
    tokens = nltk.word_tokenize(text)
    if 'RT' in tokens:
        return 1
    else:
        return 0


def number_of_mentions(text):
    return len(re.findall(r"@\S+", text))


def number_of_hashtags(text):
    return len(re.findall(r"#\S+", text))


def number_of_links(text):
    return len(re.findall(r"http\S+", text))


# Input: a text, Output: how many words are elongated
def number_of_elongated(text):
    regex = re.compile(r"(.)\1{2}")
    return len([word for word in text.split() if regex.search(word)])


# Creates a dictionary with slangs and their equivalents and replaces them
with open('slang.txt') as file:
    slang_map = dict(map(str.strip, line.partition('\t')[::2])
    for line in file if line.strip())

slang_words = sorted(slang_map, key=len, reverse=True)
regex = re.compile(r"\b({})\b".format("|".join(map(re.escape, slang_words))))
replaceSlang = partial(regex.sub, lambda m: slang_map[m.group(1)])


# Input: a text, Output: how many slang words and a list of found slangs
def number_of_slangs(text):
    slang_counter = 0
    tokens = nltk.word_tokenize(text)
    for word in tokens:
        if word in slang_words:
            slang_counter += 1
    return slang_counter


# Input: a text, Output: how many emoticons
def number_of_emoticons(text):
    return len(re.findall(r"&#\S+", text))


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
    text = re.sub(r'\&#S+', '', text)
    return text


tl, nt, irt, nom, noh, nol, noel, nos, noem = [], [], [], [], [], [], [], [], []
fkg, ari, cli, fre, gfi, lix, si, rix, dci = [], [], [], [], [], [], [], [], []
cpw, spw, wps, ttr, c, s, w, wt, lw, cw, cwdc = [], [], [], [], [], [], [], [], [], [], []

for i in range(0, len(tweets)):
    # Linguistic
    tl.append(text_length(tweets[i]))
    irt.append(is_retweet(tweets[i]))
    nom.append(number_of_mentions(tweets[i]))
    noh.append(number_of_hashtags(tweets[i]))
    nol.append(number_of_links(tweets[i]))
    noem.append(number_of_emoticons(tweets[i]))
    nt.append(number_of_tokens(tweets[i]))
    noel.append(number_of_elongated(tweets[i]))
    nos.append(number_of_slangs(tweets[i]))

    # Readability
    tweets[i] = remove_user_names(tweets[i])
    tweets[i] = remove_hashtags(tweets[i])
    tweets[i] = remove_links(tweets[i])
    tweets[i] = remove_underscore(tweets[i])
    tweets[i] = remove_emojis(tweets[i])
    measures = readability.getmeasures(tweets[i], lang='en')
    fkg.append(measures['readability grades']['Kincaid'])
    ari.append(measures['readability grades']['ARI'])
    cli.append(measures['readability grades']['Coleman-Liau'])
    fre.append(measures['readability grades']['FleschReadingEase'])
    gfi.append(measures['readability grades']['GunningFogIndex'])
    lix.append(measures['readability grades']['LIX'])
    si.append(measures['readability grades']['SMOGIndex'])
    rix.append(measures['readability grades']['RIX'])
    dci.append(measures['readability grades']['DaleChallIndex'])

    # Sentence
    cpw.append(measures['sentence info']['characters_per_word'])
    spw.append(measures['sentence info']['syll_per_word'])
    wps.append(measures['sentence info']['words_per_sentence'])
    ttr.append(measures['sentence info']['type_token_ratio'])
    c.append(measures['sentence info']['characters'])
    s.append(measures['sentence info']['syllables'])
    w.append(measures['sentence info']['words'])
    wt.append(measures['sentence info']['wordtypes'])
    lw.append(measures['sentence info']['long_words'])
    cw.append(measures['sentence info']['complex_words'])
    cwdc.append(measures['sentence info']['complex_words_dc'])

features = pd.DataFrame()
# Linguistic
features['text length'] = tl
features['number of words'] = nt
features['retweet'] = irt
features['number of mentions'] = nom
features['number of hashtags'] = noh
features['number of links'] = nol
features['number of elongated'] = noel
features['number of slangs'] = nos
features['number of emoticons'] = noem
# Readability
features['Kincaid'] = fkg
features['ARI'] = ari
features['Coleman-Liau'] = cli
features['FleschReadingEase'] = fre
features['GunningFogIndex'] = gfi
features['LIX'] = lix
features['SMOGIndex'] = si
features['RIX'] = rix
features['DaleChallIndex'] = dci
# Sentence
features['Characters per word'] = cpw
features['Syllables per word'] = spw
features['Words per sentence'] = wps
features['Type toke ratio'] = ttr
features['Characters'] = c
features['Syllables'] = s
features['Words'] = w
features['Wordtypes'] = wt
features['Long words'] = lw
features['Complex words'] = cw
features['Complex words dc'] = cwdc



'''
bn_data = bag_of_n_grams(clean_tweets, 1, 3)
tf_data = tf_idf(clean_tweets, 1, 3)
'''

