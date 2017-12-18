import features
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer


features_obj = features.Features()
df  = features_obj.get_df()

X = df['features']
y = df['Label_Major']

# vectorizer
v = DictVectorizer(sparse=False)

# print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train_vect = v.fit_transform(X_train)
X_test_vect = v.transform(X_test)

# instantiate model
logreg = LogisticRegression()

# fit model
logreg.fit(X_train_vect, y_train)

# make class predictions for the testing set
y_pred_class = logreg.predict(X_test_vect)

print(metrics.accuracy_score(y_test, y_pred_class))
print(metrics.confusion_matrix(y_test, y_pred_class))

# pickle
# vectorizer_file = open("pickled/vectorizer.pickle", "wb")
# pickle.dump(v, vectorizer_file)
# vectorizer_file.close()
#
# logreg_file = open("pickled/logreg.pickle", "wb")
# pickle.dump(logreg, logreg_file)
# logreg_file.close()