# extract and preprocess German twitter data
import os
import pandas as pd
import copy
import unicodedata
import re
import numpy as np
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import spacy

# Initializing variables
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('german')
#stopword_list.extend(['lbr', 'ja'])
nlp = spacy.load('de')

# Check the location of current working directory and move the dataset to that directory
os.getcwd()
# Or change the working directory
os.chdir('C:\\Users\\mikec\\Documents')

# Raw data
train_data = pd.read_csv("GermanTrainingData.txt", sep='\t', names=['tweet', 'coarse', 'labels'])
test_data = pd.read_csv("GermanTestingData.txt", sep='\t', names=['tweet', 'coarse', 'labels'])
data = pd.concat([train_data, test_data], ignore_index=True)
del train_data
del test_data

# Naming and dataframe convention for consistency
data = data.drop(columns="coarse")


# Changing string labels to numeric labels
def string_to_numeric(x):
    if x == 'OTHER' or x == 'PROFANITY':
        return 0
    if x == 'INSULT':
        return 1
    if x == 'ABUSE':
        return 2


data['labels'] = data['labels'].apply(string_to_numeric)

'''0: other, 1: insult, 2: abuse'''

# Create a copy for preprocess
original = copy.deepcopy(data['tweet'])
tweets = copy.deepcopy(data['tweet'])
labels = copy.deepcopy(data['labels'])


# Convert umlatus to different representations
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


# Extract only words, so remove special characters, numbers, and punctuations
def extract_only_words(text):
    text = re.sub('[^a-zA-z]', ' ', text)
    return text


# Remove stopwords (is_lower_case can be removed if the text is lower-cased before)
def remove_stopwords(text):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


# Convert the text to lower-case
def lower_case(text):
    text = text.lower()
    return text


# Remove any user names referred in the text
def remove_user_names(text):
    text = re.sub(r'@\S+', '', text)
    return text


# Remove links in the text
def remove_links(text):
    text = re.sub(r'http\S+', '', text)
    return text


# Remove white spaces in the text
def remove_white_spaces(text):
    text = " ".join(text.split())
    return text


# Lemmatize text, meaning change words to their root words; for example, been becomes be
def lemmatizing(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


# Remove underscore in the text
def remove_underscore(text):
    text = text.replace('_', '')
    return text


for i in range(0, len(tweets)-1):
    tweets[i] = remove_user_names(tweets[i])
    tweets[i] = remove_links(tweets[i])
    tweets[i] = convert_umlauts(tweets[i])
    tweets[i] = extract_only_words(tweets[i])
    tweets[i] = remove_underscore(tweets[i])
    tweets[i] = lower_case(tweets[i])
    tweets[i] = remove_white_spaces(tweets[i])
    tweets[i] = lemmatizing(tweets[i])
    tweets[i] = remove_stopwords(tweets[i])

data['tweet'] = tweets

# Split the data set into three data sets based on the labels
for labels, d in data.groupby('labels'):
    globals()['data_' + str(labels)] = d
del d

# Find the 80% cut-line for each data set
cut_0 = round(len(data_0.index) * 0.8)
cut_1 = round(len(data_1.index) * 0.8)
cut_2 = round(len(data_2.index) * 0.8)

# Construct train and test data sets
train = pd.concat([data_0.iloc[:cut_0, :], data_1.iloc[:cut_1, :], data_2.iloc[:cut_2, :]])
train = train.reindex(np.random.permutation(train.index))

test = pd.concat([data_0.iloc[cut_0:, :], data_1.iloc[cut_1:, :], data_2.iloc[cut_2:, :]])
test = test.reindex(np.random.permutation(test.index))

# Export dataframe as csv
train.to_csv("GermanCleanedTrainingData.csv", index=None, header=True, encoding='utf-8')
test.to_csv("GermanCleanedTestingData.csv", index=None, header=True, encoding='utf-8')