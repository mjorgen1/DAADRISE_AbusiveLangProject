from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# resource for understanding the Bag of Words approach
# https://machinelearningmastery.com/gentle-introduction-bag-words-model/

# get cleaned data from csv
data = pd.read_csv("")

# bag of words text to vector conversion!
# https://towardsdatascience.com/getting-your-text-data-ready-for-your-natural-language-processing-journey-744d52912867
count_vect = CountVectorizer()
english_data = count_vect.fit_transform(data['Cleaned Text'].values)