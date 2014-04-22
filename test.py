from nbrs import *

def diversity(X, y, nfold, idx_fold):
  X_train, y_train, X_test, y_test = stratifiedTrainTest(X, y, nfold, idx_fold) 
  clf.fit(X_train, y_train)
  pred_train = np.vstack([bs.predict(X_train) for bs in clf.estimators_])
  pred_test = np.vstack([bs.predict(X_test) for bs in clf.estimators_])

  es_pfm = pred_train==y_train

  nbh = NearestNeighbors().fit(X_train)
  nbrs = nbh.kneighbors(X_test, 50, return_distance=False)
  nn = nbrs[:, :20]

  #dist = DistanceMetric.get_metric('pyfunc', func=diversity_dist)
  dist = DistanceMetric.get_metric('pyfunc', func=qstats_dist)
  es = np.array([np.where(pred_test[:,i]==clf.classes_[0])+\
  np.where(pred_test[:,i]==clf.classes_[1]) for i in xrange(y_test.shape[0])])

  dist_mean = np.array([(np.mean(dist.pairwise(es_pfm[es_neg])),\
  np.mean(dist.pairwise(es_pfm[es_pos]))) for es_neg, es_pos in es])
  #dist_mean = np.array([(np.mean(dist.pairwise(es_pfm[es_neg][:, nn[i]])),\
  #np.mean(dist.pairwise(es_pfm[es_pos][:, nn[i]]))) for i, (es_neg, es_pos) in enumerate(es)])
  pred_dist = clf.classes_[np.argmin(dist_mean, axis=1)]
  #dist_mean = np.array([(np.mean(dist.pairwise(pred_train[es_neg])),\
  #np.mean(dist.pairwise(pred_train[es_pos]))) for es_neg, es_pos in es])
  #pred_dist = clf.classes_[np.argmin(dist_mean, axis=1)]
  #dist_min = np.array([(np.min(dist.pairwise(pred_train[es_neg])),\
  #np.min(dist.pairwise(pred_train[es_pos]))) for es_neg, es_pos in es])
  #pred_dist = np.argmin(dist_min, axis=1)
  print accuracy_score(y_test, pred_dist)
  return accuracy_score(y_test, pred_dist)

def affinity():
  X_train, y_train, X_test, y_test = stratifiedTrainTest(X, y, 10, 0)
  clf.fit(X_train, y_train)
  pred_train = np.vstack([bs.predict(X_train) for bs in clf.estimators_])
  dist = DistanceMetric.get_metric('pyfunc', func=dist_edit)
  ss = dist.pairwise(pred_train.T)
  af = AffinityPropagation(affinity='precomputed').fit(ss)
  cc = Counter(af.cluster_centers_indices_)
  clusters = np.array([np.where(af.labels_==i)[0] for i in xrange(af.cluster_centers_indices_.shape[0])])


#X, y = loadIris()
X, y = loadPima()
#X, y = loadBreastCancer()
clf = AdaBoostClassifier(n_estimators = n_weaks)
#adaBoost(X, y)
#pp = [diversity(X, y, 10, i) for i in xrange(10)]
