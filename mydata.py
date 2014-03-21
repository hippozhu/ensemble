import numpy as np
from sklearn import datasets, preprocessing
from sklearn.cross_validation import StratifiedKFold

def loadPima():
  data = np.loadtxt('/home/yzhu7/data/diabetes/pima-indians-diabetes.data', delimiter=',')
  X = data[:, :-1]
  X_scaled = preprocessing.MinMaxScaler().fit_transform(X)
  y = data[:, -1]
  return X_scaled, y

def loadBreastCancer():
  data = np.loadtxt('/home/yzhu7/data/uci/breast-cancer/breast-cancer-wisconsin.data.nomissing', delimiter=',')
  X = data[:, 1:-1]
  X_scaled = preprocessing.MinMaxScaler().fit_transform(X)
  y = data[:, -1]
  return X_scaled, y

def loadIris():
  iris = datasets.load_iris()
  X = iris.data[50:]
  X_scaled = preprocessing.MinMaxScaler().fit_transform(X)
  y = iris.target[50:]
  return X_scaled, y

def stratifiedTrainTest(X, y, nk, idx_fold):
  skf = StratifiedKFold(y, nk)
  idx = 0
  for train, test in skf:
    if idx == idx_fold:
      return X[train], y[train], X[test], y[test]
    idx += 1
  return None, None, None, None

