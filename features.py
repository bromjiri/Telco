import corpora
import re
import pickle
from sumy.nlp.stemmers.czech import stem_word
from sumy.utils import get_stop_words
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder

from datetime import datetime
startTime = datetime.now()


class Features:
    def __init__(self, inf_count=10000, bigram_count=50, stop=True, stem=False, bigram=False, lower=True, alpha=True):

        self.stem = stem
        self.stop = stop
        self.lower = lower
        self.alpha = alpha
        self.features = list()

        # get corpora
        # corp = corpora.Corpora("vzorek.xlsx")
        # self.df = corp.get_df(head=False)

        # get df from pickle
        df_pickled = open('pickled/corpora.pickle', "rb")
        self.df = pickle.load(df_pickled)
        df_pickled.close()

        # get unique label list
        self.label_list = self.df['obltrans_pz'].unique()

        # filter words
        self.df['filtered'] = self.df['Text'].apply(self.filter_words)
        # print(self.df['Text'].apply(self.filter_words), self.df['Text'])

        # create bestwords
        # self.create_bestwords()

        # create features list
        # for index, row in self.df.iterrows():
        #     try:
        #         feature = self.create_features(row['filtered'])
        #         self.features.append((feature, row['Label_Major']))
        #     except:
        #         pass

        # create features data frame
        self.df['features'] = self.df['filtered'].apply(self.create_features)

        # print
        # for feature in self.features:
        #     print(feature)

    def create_features(self, text):
        return dict([(word, True) for word in text])

    def get_df(self):
        return self.df

    def get_features(self):
        return self.features

    def create_bestwords(self):
        word_fd = FreqDist()
        label_word_fd = ConditionalFreqDist()
        score_fn = BigramAssocMeasures.chi_sq

        for index, row in self.df.iterrows():
            # bigram_finder = BigramCollocationFinder.from_words(row['filtered'])
            for word in row['filtered']:
                word_fd[word] += 1
                label_word_fd[row['obltrans_pz']][word] += 1
            # for bigram in bigrams:
            #     word_fd[bigram] += 1
            #     label_word_fd['pos'][bigram] += 1

        word_count = {}
        total_word_count = 0
        for label in self.label_list:
            word_count[label] = label_word_fd[label].N()
            total_word_count += label_word_fd[label].N()

        word_total_scores = {}
        for word, freq in word_fd.items():
            word_total_scores[word] = 0
            word_label_scores = {}
            for label in self.label_list:
                if label_word_fd[label][word] == 0:
                    continue

                # print(label_word_fd[label][word])
                # print(word_count[label])
                # print(total_word_count)
                word_label_scores[label] = BigramAssocMeasures.chi_sq(label_word_fd[label][word],
                                                       (freq, word_count[label]), total_word_count)
                word_total_scores[word] += word_label_scores[label]

        best = sorted(word_total_scores.items(), key=lambda tup: tup[1], reverse=True)[:1000]

        print(best)
        bestwords = set([w for w, s in best])
        self.bestwords = bestwords
        print(self.bestwords)

        print(total_word_count)
        print(word_fd['cz0035'])
        for label in self.label_list:
            print(label_word_fd[label]['cz0035'])
            print(word_count[label])

    def filter_words(self, words):

        if self.lower:
            words = Features.filter_lower(words)
        if self.stop:
            words = Features.filter_stop(words)
        if self.alpha:
            words = Features.filter_alpha(words)
        if self.stem:
            words = Features.filter_stem(words)
        return words

    @staticmethod
    def filter_lower(words):
        words_lower = list()
        for w in words:
            words_lower.append(w.lower())
        return words_lower

    @staticmethod
    def filter_stop(words):
        words_stop = list()
        for w in words:
            if w not in get_stop_words('CZECH'):
                words_stop.append(w)

        return words_stop

    @staticmethod
    def filter_alpha(words):
        words_alpha = list()
        regex = r'[a-zA-Z]+'
        for w in words:
            if re.match(regex, w):
                words_alpha.append(w)

        return words_alpha

    @staticmethod
    def filter_stem(words):
        words_stem = list()
        for w in words:
            try:
                words_stem.append(stem_word(w))
            except Exception as e:
                continue
        return words_stem


if __name__ == '__main__':

    features = Features()

    print(datetime.now() - startTime)
