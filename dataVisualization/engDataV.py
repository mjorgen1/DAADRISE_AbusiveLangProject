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
data = pd.read_csv("EnglishData.csv", encoding="utf-8")

# Label distribution
labels = copy.deepcopy(data['class'])
plt.bar(['Hate', 'Offensive', 'Neither'], [labels.value_counts()[0]/len(labels), labels.value_counts()[1]/len(labels),
                                           labels.value_counts()[2]/len(labels)])
plt.show()

# Length of tweets
lens = data.tweet.str.len()
plt.hist(lens, bins='auto')
plt.xlim(0, 300)
plt.show()

# 10 most frequently appeared words
tweets = copy.deepcopy(data['tweet'])
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

