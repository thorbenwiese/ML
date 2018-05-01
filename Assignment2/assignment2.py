# -*- coding: utf-8 -*-

import math
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from idlelib.rpc import response_queue

def generateHilbertMatrix(k):
  m = np.zeros((k,k))
  for i in range(0,k):
    for j in range(0,k):
      m[i,j] = 1.0 / float(i + j + 1)
  return m

def calculateRankAndConditionNumbers(matrix):
    print ("Rank "+ str(np.linalg.matrix_rank(matrix)))
    print ("Condition number: " + str(np.linalg.cond(matrix)))
    print ("")

def solveEquations(matrix, b):
  x = np.linalg.solve(matrix ,b)
  print (x)
  print ("Check result: " + str(np.linalg.norm(np.dot(matrix, x) - b)))
  print ("")


def readFileData():
  '''
  Format:
  0 - longitude
  1 - latitude
  2 - housing_median_age
  3 - total_rooms	total_bedrooms
  4 - population
  5 - households
  6 - median_income
  7 - median_house_value
  8 - ocean_proximity
  '''
  return np.genfromtxt("housing.csv", delimiter=',')[1:]

def findMinMaxMean(data):
  if len(data.shape) == 1:
    d = data.reshape(data.shape[0],1)
  else:
    d = data
  minVal = np.nanmin(d, axis=0)[0]
  minIdx = np.where(d.ravel()==minVal)
  maxVal = np.nanmax(d, axis=0)[0]
  maxIdx = np.where(d.ravel()==maxVal)
  mean = np.nanmean(d, axis=0)[0]

  print ("Min: " + str(minVal))
  print ("Min Indices: " + str(minIdx))
  print ("Max: " + str(maxVal))
  print ("Max Indices: " + str(maxIdx))
  print ("Mean: " + str(mean))

  return(minVal, maxVal, mean)

def euclideanDist(set1, set2):
  dist = 0
  for i in range(0,len(set1)):
    try:
        dist += pow((set1[i] - set2[i]), 2)
    except:
        pass
  return math.sqrt(dist)

def euclideanDistForTwoPoints(point1, point2):
  return math.sqrt(pow((point1 - point2), 2))

def calculateLoss(set1, set2):
  return mean_squared_error(set1, set2)
  
def neighbors(set, instance, k):
    dist = []
    for x in range(len(set)):
        dist.append([set[x], euclideanDist(instance, set[x])])
    dist.sort(key=lambda x: x[1])
    return dist[1:k] # 1 da 0 die instance selbst ist

def response(n):
    result_mean = 0
    for i in range(len(n)):
        result_mean += n[i][0][-1]
    return result_mean / len(n)
          

def knn(k, test_set):
    #test_set = test_set[0:1000]
    predictions = []
    for i in range(len(test_set)):
        n = neighbors(test_set, test_set[i], k)
        p = response(n)
        predictions.append(p) 
        #print('prediction {0:f},\t actual: {1:f}'.format(p, test_set[i][-1]))
    t = [test_set[i][-1] for i in range(len(test_set))]
    error = calculateLoss(predictions, t)
    print('error for k={0:d} is {1:f}'.format(k, error))
    return error

def main():
  # Assignment 02.2
  # a) b)
  for i in range(1,30 + 1):
    print ("k = " + str(i))
    hilbertMatrix = generateHilbertMatrix(i)
    calculateRankAndConditionNumbers(hilbertMatrix)

  # c)
  for j in [1,2,3,5,10,15,20,30,50,100]:
    print ("x_k for k = " + str(j) + ":")
    solveEquations(generateHilbertMatrix(j), np.ones(j))

  # Assignment 02.3
  # a)
  data = readFileData()
  data = data[:, ~np.isnan(data).any(axis=0)]

  # numCols is -1 because the last column ocean_proximity is only strings
  numCols = data.shape[1] - 1
  numRows = data.shape[0]

  # b)
  for i in range(0, numCols):
    findMinMaxMean(data[:,i])

  # c)
  columnNames = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "median_house_value", "ocean_proximity"]

  for i in range(0,numCols):
    minVal, maxVal, mean = findMinMaxMean(data[:,i])
    print ('MINMAXMEAN: ' + str(minVal) + ' ' + str(maxVal) + ' ' + str(mean))
    plt.figure()
    plt.title("Histogram for data column " + columnNames[i])
    plt.hist(data[:,i])
    plt.plot([],[],' ',label='Min: ' + str(minVal))
    plt.plot([],[],' ',label='Max: ' + str(maxVal))
    plt.plot([],[],' ',label='Mean: ' + str(mean))
    plt.legend()

  # d)
  plt.figure()
  plt.title("Geographical map of houses")
  plt.scatter(data[:,0], data[:,1], s=2, alpha=0.1, c=data[:,7], cmap=cm.get_cmap('RdBu_r'))

  # e)
  labels = range(0,numRows)
  values = data

  X_train, X_test, y_train, y_test = train_test_split(values, labels, test_size=0.20, random_state=42)

  findMinMaxMean(X_train)
  findMinMaxMean(X_test)

  for i in range(0,numCols):
    plt.figure()
    plt.title("Histogram for train and test data column " + columnNames[i])
    plt.hist(X_train[:,i], label='train data')
    plt.hist(X_test[:,i], label='test data')
    plt.legend(loc='upper right')
  



  # Assignment 02.4
  # a)
  np.random.seed(9876)
  mod = GradientBoostingRegressor(loss='ls')
  fit = mod.fit(X_train, y_train)
  predict = fit.predict(X_test)
  #print predict
  #print y_test
  loss = calculateLoss(predict, y_test)
  # TODO loss ist viel zu hoch?!
  print(loss)

  # b)
  distance = euclideanDist(data[:,0], data[:,1])
  # TODO distance ist viel zu gross?!
  # vielleicht sollte für kNN die Funktion euclideanDistForTwoPoints()
  # verwendet werden..
  print ('Distance: ' + repr(distance))
  

  plt.figure()
  err = []
  for k in range(2,20):
    err.append(knn(k, X_train[:100]))
  plt.plot(err)
  
  
    # plots all figures
  plt.show()

  # c)
  # implement kNN
  # siehe https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/


  # d)
  # predict housing values for different values of k
  # what are training and test errors?

# execute main
if __name__ == "__main__":
  main()
