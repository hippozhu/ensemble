import numpy as np
import math
from mydata import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cross_validation import StratifiedKFold

def adaBoost(X, y, n_weaks = 100):
  clf = AdaBoostClassifier(n_estimators = n_weaks)
  score = cross_val_score(clf, X, y, cv=StratifiedKFold(y, nfold))
  print score
  print score.mean()

def local_ensemble(X, y, idx_fold = 0, k=10):
  # train model and obtain performance on both train and test
  X_train, y_train, X_test, y_test = stratifiedTrainTest(X, y, nfold, idx_fold)
  clf = AdaBoostClassifier(n_estimators = n_weaks)
  clf.fit(X_train, y_train)
  y_pred_es_test = np.vstack([es.predict(X_test) for es in clf.estimators_])
  hits_test= np.array([y_test[i]==y_pred_es_test[:,i] for i in xrange(len(y_test))])
  y_pred_es_train =  np.vstack([es.predict(X_train)  for es in clf.estimators_])
  hits_train = np.array([y_train[i]==y_pred_es_train[:,i] for i in xrange(len(y_train))])

  # get neiborhood and neighbors
  nbh = NearestNeighbors().fit(X_train)
  nbrs = nbh.kneighbors(X_test, 50, return_distance=False)
  
  # select local expert estimators
  experts = min_hits_nbh(k, nbrs, hits_train)

  # classify with experts
  y_test_pred_expert = expert_classify_proba(clf.estimators_, experts, X_test)
  return accuracy_score(y_test, y_test_pred_expert)

def expert_classify_proba(base_classifiers, experts, X_test):
  probas = np.vstack(np.mean([base_classifiers[j].predict_proba(X_test[i])[0] for j in np.where(experts[i] == True)[0]], axis = 0) for i in xrange(X_test.shape[0]))
  labels = base_classifiers[0].classes_
  return labels[np.argmax(probas, axis = 1)] 
  
def select_by_local_precision(base_clfs, nbrs, k):
  nn = nbrs[:, :k]
  [np.sum(expert_train[nni], axis = 0) for nni in nn]
#def expert_classify_voting(base_classifiers, experts, X_test):

def neighborhood_good_es(X, y, idx_fold, k):
  X_train, y_train, X_test, y_test = stratifiedTrainTest(X, y, nfold, idx_fold)
  clf = AdaBoostClassifier(n_estimators = n_weaks)
  clf.fit(X_train, y_train)
  y_pred_es_test = np.vstack([es.predict(X_test) for es in clf.estimators_])
  hits_test= np.array([y_test[i]==y_pred_es_test[:,i] for i in xrange(len(y_test))])

  y_pred_es_train =  np.vstack([es.predict(X_train)  for es in clf.estimators_])
  hits_train = np.array([y_train[i]==y_pred_es_train[:,i] for i in xrange(len(y_train))])

  nbh = NearestNeighbors().fit(X_train)
  nbrs = nbh.kneighbors(X_test, 50, return_distance=False)

  hits_nbh = common_good_es_nbh(k, nbrs, hits_train)
  #hits_nbh = min_hits_nbh(k, nbrs, hits_train)
  hits_both = np.logical_and(hits_test, hits_nbh)
  return 1 - np.sum(np.divide(np.asarray(np.sum(hits_both, axis=1), dtype=np.float32), np.sum(hits_nbh, axis=1)) <= 0.5)/float(y_test.shape[0])

# select common good classifier shared by k closest neighbors
# if none, select all
def common_good_es_nbh(k, nbrs, hits_train):
  nn = nbrs[:,:k]
  common_es = np.array([np.all(hits_train[nni], axis=0) for nni in nn])
  no_commons = np.where(np.sum(common_es, axis=1)==0)[0]
  common_es[no_commons] = np.array([True] * common_es.shape[1])
  return common_es
  
# select common good classifier shared by k closest neighbors
# if none, select those shared by k-1, k-2 etc.
def min_hits_nbh(k, nbrs, hits_train, min_threshold = 0):
  update = np.array(xrange(nbrs.shape[0]))
  hits = hits_train[nbrs[:,0]]
  min_hits = hits.copy()
  for i in xrange(1, k):
    hits = np.logical_and(hits, hits_train[nbrs[:,i]])
    update = np.where(np.sum(hits, axis=1)>min_threshold)[0]
    if update.shape[0] == 0:
      break
    min_hits[update] = hits[update]
  return min_hits
    
#local accuracy selection + meta learning with local accuracy feature
#def local_accuracy_nbh():

#cross_val_score(clf, X, y, cv=StratifiedKFold(y, 5))
#cross validation accuracy:
#Ada: [ 0.71428571, 0.81168831,  0.73376623,  0.73202614,  0.77777778] 0.7539 +- 0.0357
#k=2: [ 0.75974025, 0.67532467,  0.75324675,  0.68831168,  0.79220779] 0.7337 +- 0.0446

n_weaks=100
nfold = 10
if __name__ == "__main__":
  #X, y = loadIris()
  X, y = loadBreastCancer()
  adaBoost(X, y)
  #X, y = loadPima()
'''
X_train, y_train, X_test, y_test = stratifiedTrainTest(X, y, 10, 0)
clf = AdaBoostClassifier(n_estimators = n_weaks)
clf.fit(X_train, y_train)
y_pred_es_test = np.vstack([es.predict(X_test) for es in clf.estimators_])
expert_test= np.array([y_test[i]==y_pred_es_test[:,i] for i in xrange(len(y_test))])
y_pred_es_train =  np.vstack([es.predict(X_train)  for es in clf.estimators_])
expert_train = np.array([y_train[i]==y_pred_es_train[:,i] for i in xrange(len(y_train))])
nbh = NearestNeighbors().fit(X_train)
nbrs = nbh.kneighbors(X_test, 50, return_distance=False)
'''
