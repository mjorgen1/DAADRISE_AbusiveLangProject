import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import os
import copy

pd.options.display.max_colwidth = 200

#Or change the working directory
os.chdir('C:\\Users\\mikec\\Documents')

# Raw data
data = pd.read_csv("EnglishData.csv")

# Remove first five columns, since they are not needed for the AutoML
data = data.iloc[:, 5:]

# Naming and dataframe convention for consistency
data.rename(columns={'class': 'labels'}, inplace=True)
data = data[['tweet', 'labels']]

'''0: hate speech, 1: offensive language, 2: neither'''

tweets = copy.deepcopy(data['tweet'])
labels = copy.deepcopy(data['labels'])

