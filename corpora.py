import pandas as pd
from bs4 import BeautifulSoup
import nltk
import pickle


class Corpora:
    def __init__(self, input_file_path):
        self.df = pd.read_excel(input_file_path, sheet_name='zdroj', usecols='B,E,F,H')
        self.extract_text()
        self.extract_label()
        # print(self.df.head())

    def extract_text(self):
        self.df['Text'] = self.df.apply(self.get_words, axis=1)
        # print(self.df['Text'].head())

    def extract_label(self):
        self.df['Label_Major'] = self.df.apply(self.get_major, axis=1)

    def get_df(self):
        return self.df

    def get_major(self, row):
        print(row['obltrans_pz'])
        try:
            return row['obltrans_pz'].split("-")[0].strip()
        except:
            return ''

    def get_words(self, row):

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

    print(df['Text'][56])
    print(df['obltrans_pz'][56])