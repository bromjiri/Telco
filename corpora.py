import pandas as pd
from bs4 import BeautifulSoup
import nltk
import pickle


class Corpora:

    def __init__(self, input_file_path):
        self.df = pd.read_excel(input_file_path, sheet_name='zdroj', usecols='B,E,F,H')
        # print(self.df.head())

    def get_df(self, head=False):
        self.df['Text'] = self.df.apply(get_words, axis=1)
        # print(self.df['Text'].head())
        if head:
            return self.df.head()
        else:
            return self.df


def get_words(row):

    # define output list
    unigrams = list()

    # parse e-mail subject
    for token in nltk.word_tokenize(str(row['Subject'])):
        unigrams.append(token)

    # parse e-mail body
    parsed_content = BeautifulSoup(row['popis'], 'html.parser')
    paragraphs = parsed_content.find_all('p')
    for p in paragraphs:
        for token in nltk.word_tokenize(p.text):
            unigrams.append(token)
    # print(unigrams)

    return unigrams


if __name__ == '__main__':
    # corp = Corpora("vzorek.xlsx")
    # df = corp.get_df()
    # print(df.head())

    # pickle
    # df_file = open("pickled/corpora.pickle", "wb")
    # pickle.dump(df, df_file)
    # df_file.close()

    # unpickle
    df_pickled = open('pickled/corpora.pickle', "rb")
    df = pickle.load(df_pickled)
    df_pickled.close()

    print(df.head())