import numpy as np
import math

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import NearestNeighbors

n_weaks=100
X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)

X_test, y_test = X[2000:], y[2000:]
X_train, y_train = X[:2000], y[:2000]

clf = AdaBoostClassifier(n_estimators = n_weaks)
cross_val_score(clf, X_train, y_train)
clf.fit(X_train, y_train)
errs = [1.0-es.score(X_train, y_train) for es in clf.estimators_]
alpha = [math.log((1-e)/e) for e in errs]
#alpha = [1.] * n_weaks
y_pred_es = np.vstack([es.predict(X_test) for es in clf.estimators_])
hit_rate = np.array([sum(y_test[i]==y_pred_es[:,i]) for i in xrange(len(y_test)])
np.histogram(hit_rate, range=(0,100))

y_pred_es_train =  np.vstack([es.predict(X_train)  for es in clf.estimators_])
hits_train = np.array([y_train[i]==y_pre_es_train[:,i] for i in xrange(len(y_train))])
nbh = NearestNeighbors().fit(X_train)
nbrs = nbh.kneighbors(X_test, 100, return_distance=False)
'''
pred = [es.predict(X_test) for es in clf.estimators_]
pred = [np.multiply(alpha[i], pred[i]) for i in xrange(n_weaks)]
avg_pred = np.array([-1. if x<0 else 1. for x in sum(pred)])
pred_proba = [es.predict_proba(X_test) for es in clf.estimators_]
pred_proba = [np.multiply(alpha[i], pred_proba[i]) for i in xrange(n_weaks)]
avg_pred = np.array([1. if x[0]<x[1] else -1. for x in sum(pred_proba)])
print(confusion_matrix(y_test, avg_pred))
print(accuracy_score(y_test, avg_pred))
'''
#print(classification_report(y_test, avg_pred))
#alpha = [1.] * n_weaks
#pred = [es.predict(X_test) for es in clf.estimators_]
#avg_pred = np.array([1. if x>0 else -1. for x in sum(pred)])

