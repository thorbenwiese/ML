# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
from sklearn.metrics import zero_one_loss

'''
05.1.a)
'''
def loadMatFile():
  data = sio.loadmat('iris_multiclass.mat')

  meas = data['meas']
  species = data['species']
  testIndices = data['indices_test']
  trainIndices = data['indices_train']

  train_data = []
  test_data = []

  for idx in trainIndices[0]:
    train_data.append([meas[idx - 1], species[idx - 1]])
  for idx in testIndices[0]:
    test_data.append([meas[idx - 1], species[idx - 1]])

  test_labels = []
  for element in test_data:
    c = element[1][0]
    if c == 'setosa':
      test_labels.append([1])
    elif c == 'versicolor':
      test_labels.append([2])
    elif c == 'virginica':
      test_labels.append([3])

  return train_data, test_data, test_labels

def least_squares(X, y):
  return np.linalg.lstsq(X,y)[0]

'''
classNum:
0 setosa
1 versicolor
2 virginica
'''
def extract_classes(data, classNum):
  result = []
  labels = []

  for d in data:
    if d[1] == 'setosa' and classNum == 0:
      result.append(d[0])
      labels.append(1)
    elif d[1] == 'versicolor' and classNum == 1:
      result.append(d[0])
      labels.append(1)
    elif d[1] == 'virginica' and classNum == 2:
      result.append(d[0])
      labels.append(1)
    else:
      result.append(d[0])
      labels.append(0)

  return result, labels

def zeroOneLoss(y, y_pred):
  print '0/1-Loss: ', zero_one_loss(y, y_pred, False), '(', zero_one_loss(y, y_pred), ' %)'
  '''
  loss = 0
  for i in range(len(y)):
    if y[i] != y_pred[i]:
      loss = loss + 1
  return loss
  '''

def predict_classes(w1, w2, w3, test_data):
  result = []

  p1 = np.poly1d(w1)
  p2 = np.poly1d(w2)
  p3 = np.poly1d(w3)

  for i in range(len(test_data)):
    m = np.max([p1(i), p2(i), p3(i)])
    if p1(i) == m:
      result.append([1])
    elif p2(i) == m:
      result.append([2])
    elif p3(i) == m:
      result.append([3])

  return result

def main():
  train_data, test_data, test_labels = loadMatFile()

  setosa, setosa_labels = extract_classes(train_data, 0)
  versicolor, versicolor_labels = extract_classes(train_data, 1)
  virginica, virginica_labels = extract_classes(train_data, 2)

  w_setosa = least_squares(setosa, setosa_labels)
  w_versicolor = least_squares(versicolor, versicolor_labels)
  w_virginica = least_squares(virginica, virginica_labels)

  print '\nWeight vectors:'
  print 'w_setosa: ', w_setosa
  print 'w_versicolor: ', w_versicolor
  print 'w_virginica: ', w_virginica
  print ''

  prediction = predict_classes(w_setosa, w_versicolor, w_virginica, test_labels)

  print 'Actual: \n', test_labels, '\n'
  print 'Prediction: \n', prediction, '\n'
  zeroOneLoss(prediction, test_labels)

  '''
  c)

  What are potential problems of using least-squares regressions for 
  multi-class classification?

  - scale of confidence values may differ between classifiers
  - OvR sees negative values as majority -> distribution unbalanced
  '''

# execute main
if __name__ == "__main__":
  main()
