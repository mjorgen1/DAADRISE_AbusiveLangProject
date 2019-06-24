# extract English twitter data here
import pandas as pd
import os

#Check the location of current working directory and move the dataset to that directory
os.getcwd()
#Or change the working directory
os.chdir('C:\\Users\\mikec\\Documents')


# Raw data
data = pd.read_csv("EnglishData.csv")

# The first column is removed, since it just a sequence of numbers from 1 to 25296
data = data.drop(columns="Unnamed: 0")

# Playing
print(data['class'].value_counts()/data['class'].size)



