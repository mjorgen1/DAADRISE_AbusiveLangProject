import pandas as pd
import copy
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import nltk


# TODO: to use stopwords below, you may need to run the below in your python env
# import nltk
# nltk.download('stopwords')


# Load data from csv
# TODO: adjust the path as need be and use the csv called "EnglishCleanedData.csv" in dataCleaning folder
# the first column is the cleaned tweet
# the second column is the original tweet
# the third column is the labeled tweet
# 0: hate speech, 1: offensive language, 2: neither
data = pd.read_csv("/home/mackenzie/git_repositories/DAADRISE_AbusiveLangProject/dataCleaning/EnglishCleanedData.csv", encoding="utf-8")

# Figure 1: Label distribution
labels = copy.deepcopy(data['labels'])
plt.bar(['Hate', 'Offensive', 'Neither'], [labels.value_counts()[0]/len(labels), labels.value_counts()[1]/len(labels),
                                           labels.value_counts()[2]/len(labels)])
plt.show()
print('Hate speech count: ')
print(labels.value_counts()[0]/len(labels))
print('Offensive speech count: ')
print(labels.value_counts()[1]/len(labels))
print('Neither speech count: ')
print(labels.value_counts()[2]/len(labels))

# Figure 2: Length of tweets
# TODO: you can change the tweet column name below: "cleaned_tweet" or "tweet" (original tweet)
lens = data.cleaned_tweet.str.len()
plt.hist(lens, bins='auto')
plt.xlim(0, 300)
plt.show()

# Figure 3: 10 most frequently appeared words
tweets = copy.deepcopy(data['cleaned_tweet'])
for i in range(0, len(tweets)-1):
    currLine = tweets[i]
    newLine = ""
    newLine = re.sub('[^a-zA-Z]', ' ', currLine)
    newLine = re.sub(r'\s+', ' ', newLine)
    newLine = newLine.lower()
    tweets[i] = newLine

unique_frequencies = dict()
total_frequencies = dict()

stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(['rt', 'co', 'http', 'u', 'got', 'get'])

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
l, ys = zip(*high)
xs = np.arange(len(l))
width = 0.5
plt.bar(xs, ys, width, align='center')
plt.xticks(xs, l)
plt.show()

