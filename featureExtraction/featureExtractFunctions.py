import re
import nltk
from functools import partial


def text_length(text):
    return len(text)


def number_of_tokens(text):
    tokens = nltk.word_tokenize(text)
    return len(tokens)


def is_retweet(text):
    tokens = nltk.word_tokenize(text)
    if 'RT' in tokens:
        return 1
    else:
        return 0


def number_of_mentions(text):
    return len(re.findall(r"@\S+", text))


def number_of_hashtags(text):
    return len(re.findall(r"#\S+", text))


def count_multi_exclamation_marks(text):
    return len(re.findall(r"(\!)\1+", text))


# Counts number of multi stop marks
def count_multi_question_marks(text):
    return len(re.findall(r"(\?)\1+", text))


# Counts number of multi stop marks
def count_multi_stop_marks(text):
    return len(re.findall(r"(\.)\1+", text))


# Input: a text, Output: how many words are elongated
def count_elongated(text):
    regex = re.compile(r"(.)\1{2}")
    return len([word for word in text.split() if regex.search(word)])


# Input: a text, Output: how many words are all caps
def count_all_caps(text):
    return len(re.findall("[A-Z0-9]{3,}", text))


# Creates a dictionary with slangs and their equivalents and replaces them
with open('slang.txt') as file:
    slang_map = dict(map(str.strip, line.partition('\t')[::2])
    for line in file if line.strip())

slang_words = sorted(slang_map, key=len, reverse=True)
regex = re.compile(r"\b({})\b".format("|".join(map(re.escape, slang_words))))
replaceSlang = partial(regex.sub, lambda m: slang_map[m.group(1)])


# Input: a text, Output: how many slang words and a list of found slangs
def count_slang(text):
    slang_counter = 0
    slangs_found = []
    tokens = nltk.word_tokenize(text)
    for word in tokens:
        if word in slang_words:
            slangs_found.append(word)
            slang_counter += 1
    return slang_counter, slangs_found


# Input: a text, Output: how many emoticons
def count_emoticons(text):
    return len(re.findall(r"(\&#)\1+", text))

