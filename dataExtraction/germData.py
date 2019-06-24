# extract German Twitter data here from txt file and save to new excel sheet
import pandas as pd
import os

#Check the location of current working directory and move the dataset to that directory
os.getcwd()
#Or change the working directory
os.chdir('C:\\Users\\mikec\\Documents')

# Raw data
train_data = pd.read_csv("GermanTrainingData.txt", sep='\t', names=['tweets', 'coarse', 'fine'])
test_data = pd.read_csv("GermanTrainingData.txt", sep='\t', names=['tweets', 'coarse', 'fine'])