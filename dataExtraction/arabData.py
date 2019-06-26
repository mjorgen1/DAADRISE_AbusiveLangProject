# extract Arabic twitter data here

import pandas as pd
import os
import numpy as np

#Check the location of current working directory and move the dataset to that directory
os.getcwd()
#Or change the working directory
os.chdir('C:\\Users\\mikec\\Documents')

# Raw data
data = pd.read_excel("ArabicData.xlsx")

# Remove first five columns, since they are not needed for the AutoML
data = data.iloc[:, 2:4]

# Naming and dataframe convention for consistency
data.rename(columns={'text': 'tweet', 'aggregatedAnnotation': 'labels'}, inplace=True)
data['labels'] = data['labels'].abs()

'''0: clean, 1: offensive, 2: obscene'''

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
train.to_csv("ArabicCleanedTrainingData.csv", index=None, header=True)
test.to_csv("ArabicCleanedTestingData.csv", index=None, header=True)