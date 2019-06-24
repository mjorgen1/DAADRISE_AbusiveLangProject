# extract English twitter data here
import pandas as pd
import numpy as np

data = pd.read_csv("EnglishData.csv");
data = data.drop(columns="Unnamed: 0");

print(data['class'].value_counts()/data['class'].size);




