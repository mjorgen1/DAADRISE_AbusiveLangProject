import pandas as pd
import os
import copy
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import nltk

#Check the location of current working directory and move the dataset to that directory
os.getcwd()
#Or change the working directory
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

# Length of tweets
lens = data.tweet.str.len()
plt.hist(lens, bins='auto')
plt.show()


# 10 most frequently appeared words
tweets = copy.deepcopy(data['tweet'])
for i in range(0, len(tweets)-1):
    currLine = tweets[i]
    newLine = ""
    newLine = re.sub('[^a-üA-Ü]', ' ', currLine)
    newLine = re.sub(r'\s+', ' ', newLine)
    newLine = newLine.lower()
    tweets[i] = newLine

unique_frequencies = dict()
total_frequencies = dict()

stopwords = nltk.corpus.stopwords.words('german')
stopwords.extend(['|lbr|', 'ja'])
#'mehr', 'mal', 'immer', 'gibt', 'd', 'u'

for readme in tweets:
    words = nltk.word_tokenize(readme)
    fdist = nltk.FreqDist(words)
    for word, freq in fdist.most_common(50):
        if word not in stopwords:
            if word not in total_frequencies.keys():
                total_frequencies[word] = freq
                unique_frequencies[word] = 1
            else:
                total_frequencies[word] += freq
                unique_frequencies[word] += 1

k = Counter(total_frequencies)

high = k.most_common(10)
labels, ys = zip(*high)
xs = np.arange(len(labels))
width = 0.5
plt.bar(xs, ys, width, align='center')
plt.xticks(xs, labels, rotation = 45)
plt.show()