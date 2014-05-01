import numpy as np
from sklearn import datasets, preprocessing
from sklearn.cross_validation import StratifiedKFold

dataset_name = None

def loadPima():
  dataset_name = 'Pima'
  data = np.loadtxt('/home/yzhu7/data/diabetes/pima-indians-diabetes.data', delimiter=',')
  X = data[:, :-1]
  X_scaled = preprocessing.MinMaxScaler().fit_transform(X)
  y = data[:, -1].astype(np.int32)
  return X_scaled, y

def loadBreastCancer():
  dataset_name = 'Breast Cancer'
  data = np.loadtxt('/home/yzhu7/data/uci/breast-cancer/breast-cancer-wisconsin.data.nomissing', delimiter=',')
  X = data[:, 1:-1]
  X_scaled = preprocessing.MinMaxScaler().fit_transform(X)
  y = data[:, -1].astype(np.int32)
  return X_scaled, y

def loadIonosphere():
  dataset_name = 'Ionosphere'
  data = np.loadtxt('/home/yzhu7/data/uci/ionosphere/ionosphere.data', delimiter=',')
  X = data[:, 1:-1]
  X_scaled = preprocessing.MinMaxScaler().fit_transform(X)
  y = data[:, -1].astype(np.int32)
  return X_scaled, y

def loadIris():
  dataset_name = 'Iris'
  iris = datasets.load_iris()
  X = iris.data[50:]
  X_scaled = preprocessing.MinMaxScaler().fit_transform(X)
  y = iris.target[50:].astype(np.int32)
  return X_scaled, y

def stratifiedTrainTest(X, y, nk, idx_fold):
  skf = StratifiedKFold(y, nk)
  idx = 0
  for train, test in skf:
    if idx == idx_fold:
      return X[train], y[train], X[test], y[test]
    idx += 1
  return None, None, None, None

