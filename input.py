import pandas as pd
from bs4 import BeautifulSoup
import nltk
from sklearn.naive_bayes import MultinomialNB
from nltk.classify.scikitlearn import SklearnClassifier
from sumy.nlp.stemmers.czech import stem_word
from sumy.utils import get_stop_words
import random
import re
import numpy as np

# get body from html
def get_body(source_code):
    unigrams = list()
    parsed_content = BeautifulSoup(source_code, 'html.parser')
    paragraphs = parsed_content.find_all('p')
    for p in paragraphs:
        for token in nltk.word_tokenize(p.text):
            unigrams.append(token)
    # print(unigrams)
    return unigrams


# create dictionary (bag of words)
def create_feature(words):
    words = filter_lower(words)
    words = filter_alpha(words)
    words = filter_stop(words)
    words = filter_stem(words)
    return dict([(word, True) for word in words])


def filter_lower(words):
    words_lower = list()
    for w in words:
        words_lower.append(w.lower())
    return words_lower


def filter_stop(words):
    words_stop = list()
    for w in words:
        if w not in get_stop_words('CZECH'):
            words_stop.append(w)

    return words_stop

def filter_alpha(words):
    words_alpha = list()
    regex = r'[a-zA-Z]+'
    for w in words:
        if re.match(regex, w):
            words_alpha.append(w)

    return words_alpha


def filter_stem(words):
    words_stem = list()
    for w in words:
        try:
            words_stem.append(stem_word(w))
        except:
            continue
    return words_stem

# START	
# read input
input_file_path = 'vzorek.xlsx'
df = pd.read_excel(input_file_path, sheet_name='zdroj')

features = list()
# creating features
for index, row in df.iterrows():
    words = get_body(df.at[index, 'popis'])
    feature = create_feature(words)

    features.append((feature, df.at[index, 'obltrans_pz']))
    # if df.at[index, 'produkt'] == np.nan:
    #     continue
    #
    # features.append((feature, df.at[index, 'produkt']))

# for feature in features:
#     print(feature)
# print(len(features))

random.shuffle(features)
training_set = features[:2100]
testing_set = features[2100:]


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
mnb = (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100
mnb = round(mnb, 1)
print(mnb)