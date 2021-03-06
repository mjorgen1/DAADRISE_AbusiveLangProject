# extract and preprocess English twitter data
import os
import pandas as pd
import copy
import unicodedata
import re
import numpy as np
import nltk
import spacy
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
# need to run the following once:
# import nltk
# nltk.download('stopwords')

# run the below in terminal:
# python3 -m spacy download en_core_web_md


''' Initializing variables '''
tokenizer = ToktokTokenizer()
stopword_list = stopwords.words('english')
stopword_list.extend(['rt'])
nlp = spacy.load('en_core_web_md')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

''' Raw Data'''
# put in path for dataset
data = pd.read_csv("labeled_data.csv")

# Remove first five columns, since they are not needed for the AutoML
data = data.iloc[:, 5:]

# Naming and dataframe convention for consistency
data.rename(columns={'class': 'labels'}, inplace=True)
data = data[['tweet', 'labels']]

'''0: hate speech, 1: offensive language, 2: neither'''


# Create a copy for preprocess
tweets = copy.deepcopy(data['tweet'])
labels = copy.deepcopy(data['labels'])


''' Data Cleaning Functions'''
# Remove accented characters in the text
def remove_accent(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


# Contraction map for expanding contractions
CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}


# Expand contractions; for example, don't becomes do not
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


# Extract only words, so remove special characters, numbers, and punctuations
def extract_only_words(text):
    text = re.sub(r'[^\w\s]', ' ', text)
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


# Remove hashtags
def remove_hashtags(text):
    text = re.sub(r'#\S+', '', text)
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


# Stem text, similar to lemmatizing instead stem can lead to random words
def stemming(text):
    token_words = word_tokenize(text)
    filtered_text = []
    for word in token_words:
        filtered_text.append(stemmer.stem(word))
        filtered_text.append(" ")
    return "".join(filtered_text)


# Remove underscore in the text
def remove_underscore(text):
    text = text.replace('_', '')
    return text


# Replace an elongated word with its basic form, unless the word exists in the lexicon
def replace_elongated(text):
    filtered_tokens = []
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    for w in tagged:
        word = w[0]
        repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        repl = r'\1\2\3'
        if wordnet.synsets(word):
            final_word = word
        else:
            repl_word = repeat_regexp.sub(repl, word)
            if repl_word != word:
                final_word = replace_elongated(repl_word)
            else:
                final_word = repl_word

        filtered_tokens.append(final_word)

    filtered_text = " ".join(filtered_tokens)
    return filtered_text


for i in range(0, len(tweets)):
    tweets[i] = remove_user_names(tweets[i])
    tweets[i] = remove_hashtags(tweets[i])
    tweets[i] = remove_links(tweets[i])
    tweets[i] = expand_contractions(tweets[i])
    tweets[i] = remove_accent(tweets[i])
    tweets[i] = extract_only_words(tweets[i])
    tweets[i] = remove_underscore(tweets[i])
    tweets[i] = lower_case(tweets[i])
    tweets[i] = remove_white_spaces(tweets[i])
    tweets[i] = replace_elongated(tweets[i])
    tweets[i] = lemmatizing(tweets[i])
    tweets[i] = stemming(tweets[i])
    tweets[i] = remove_stopwords(tweets[i])

data = pd.concat([tweets, data], axis=1)
data.columns = ['cleaned_tweet', 'tweet', 'labels']
data['cleaned_tweet'].replace('', np.nan, inplace=True)
data.dropna(subset=['cleaned_tweet'], inplace=True)


# Export dataframe as csv
data.to_csv("EnglishCleanedData.csv", index=None, header=True, encoding='utf-8')

# Option to export dataframe as tsv
data.to_csv("labeled_tweets.tsv", index=True, header=False, sep="\t", encoding='utf-8')

# Other options below:
# Drop cleaned tweet, if only want original tweet
#data=data.drop('cleaned_tweet', axis=1)
'''0: hate speech, 1: offensive language, 2: neither'''
# Change labels back to string class form
#data['labels']= data['labels'].replace({0: 'hate_speech', 1: 'offensive_language', 2: 'neither'})



