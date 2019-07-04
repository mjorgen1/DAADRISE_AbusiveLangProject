import re
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample sentence
s = "one two three four five"
s = s.lower()
s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s) # lowercases all uppercase words, takes out punctuation, and allows for numbers
tokens = [token for token in s.split(" ") if token != ""]
output = list(ngrams(tokens, 3)) # the higher the number the smaller the ngrams list
print(output)

# now use tfidf on multiple ngrams to get the weights of the different ngrams
vectorizer = TfidfVectorizer()
# it's expecting a whole document so may need more strings?
got_tfidf = vectorizer.fit_transform(output) # the input for tfidf would be the ngrams from above
tfidf = pd.DataFrame(got_tfidf.toarray())
tfidf.columns = vectorizer.get_feature_names()