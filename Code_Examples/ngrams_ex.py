import re
from nltk.util import ngrams

# Sample sentence
s = "one two three four five"
s = s.lower()
s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s) # lowercases all uppercase words, takes out punctuation, and allows for numbers
tokens = [token for token in s.split(" ") if token != ""]
output = list(ngrams(tokens, 5)) # the higher the number the smaller the ngrams list
print(output)