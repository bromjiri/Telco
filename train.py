import nltk
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, NuSVC
from nltk.classify.scikitlearn import SklearnClassifier
import random
import features

features_obj = features.Features()
features_list = features_obj.get_features()

random.shuffle(features_list)
training_set = features_list[:2100]
testing_set = features_list[2100:]


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
mnb = (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100
mnb = round(mnb, 1)
print(mnb)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
# BernoulliNB_classifier._vectorizer.sort = False
BernoulliNB_classifier.train(training_set)
bnb = (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100
bnb = round(bnb, 1)
print(bnb)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
# LogisticRegression_classifier._vectorizer.sort = False
LogisticRegression_classifier.train(training_set)
lr = (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100
lr = round(lr, 1)
print(lr)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
# LinearSVC_classifier._vectorizer.sort = False
LinearSVC_classifier.train(training_set)
lsvc = (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100
lsvc = round(lsvc, 1)
print(lsvc)
