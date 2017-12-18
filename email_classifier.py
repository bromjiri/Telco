import pickle
import re
import nltk
from sumy.utils import get_stop_words
from sumy.nlp.stemmers.czech import stem_word
import sys


class Features:
    def __init__(self, email, stop=True, stem=False, lower=True, alpha=True):

        self.stem = stem
        self.stop = stop
        self.lower = lower
        self.alpha = alpha
        self.features = list()

        words = nltk.word_tokenize(email)

        filtered = self.filter_words(words)

        self.features = self.create_features(filtered)

    def create_features(self, text):
        return dict([(word, True) for word in text])

    def get_features(self):
        return self.features

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

    # unpickle vectorizer
    vectorizer_pickled = open('pickled/vectorizer.pickle', "rb")
    vectorizer = pickle.load(vectorizer_pickled)
    vectorizer_pickled.close()

    # unpickle logreg
    logreg_pickled = open('pickled/logreg.pickle', "rb")
    logreg = pickle.load(logreg_pickled)
    logreg_pickled.close()

    try:
        print("input file: " + str(sys.argv[1]))
    except:
        print("Missing argument: input file name")
        exit()

    # input_file_path = "input_emails.txt"
    input_file_path = sys.argv[1]
    input_file = open(input_file_path, 'r', encoding='utf8')

    for line in input_file:

        print("\n*************")
        print(line.strip())

        # sentence = "Dobrý den, prosím o změnu tarifu."
        # sentence = "Dobrý den, prosím o zrušení linky."

        features = Features(line)
        feature = features.get_features()

        # print(feature)

        vector = vectorizer.transform(feature)

        label = logreg.predict(vector)
        print("predicted class: " + str(label))

        print("all classes: " + str(logreg.classes_))
        probs = logreg.predict_proba(vector)
        print("probabilities: " + str(probs))