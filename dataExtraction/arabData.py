# extract Arabic twitter data here

import pandas as pd
import os
import xlrd

#Check the location of current working directory and move the dataset to that directory
os.getcwd()
#Or change the working directory
os.chdir('C:\\Users\\mikec\\Documents')

# Raw data
data = pd.read_excel("ArabicData.xlsx")

# The first column is removed, since it just a sequence of numbers from 1 to 25296
data = data.drop(columns="#")