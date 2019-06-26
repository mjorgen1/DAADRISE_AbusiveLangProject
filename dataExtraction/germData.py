# extract German Twitter data here from txt file and save to new excel sheet
import pandas as pd
import os
import numpy as np

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

'''0: other, 1: insult, 2: abuse'''

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
train.to_csv("GermanCleanedTrainingData.csv", index=None, header=True)
test.to_csv("GermanCleanedTestingData.csv", index=None, header=True)