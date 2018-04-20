import math
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

def generateHilbertMatrix(k):
  m = np.zeros((k,k))
  for i in range(0,k):
    for j in range(0,k):
      m[i,j] = 1.0 / float(i + j + 1)
  return m

def calculateRankAndConditionNumbers(matrix):
  print "Rank: " + str(np.linalg.matrix_rank(matrix))
  print "Condition number: " + str(np.linalg.cond(matrix))
  print ""

def solveEquations(matrix, b):
  x = np.linalg.solve(matrix ,b)
  print x
  print "Check result: " + str(np.linalg.norm(np.dot(matrix, x) - b))
  print ""


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
  minVal = np.nanmin(data, axis=0)[0]
  minIdx = np.where(data.ravel()==minVal)
  maxVal = np.nanmax(data, axis=0)[0]
  maxIdx = np.where(data.ravel()==maxVal)
  mean = np.nanmean(data, axis=0)[0]

  print "Min: " + str(minVal)
  print "Min Indices: " + str(minIdx)
  print "Max: " + str(maxVal)
  print "Max Indices: " + str(maxIdx)
  print "Mean: " + str(mean)

def euclideanDist(set1, set2):
  dist = 0
  for i in range(0,len(set1)):
    dist += pow((set1[i] - set2[i]), 2)
  return math.sqrt(dist)

def euclideanDistForTwoPoints(point1, point2):
  return math.sqrt(pow((point1 - point2), 2))

def calculateLoss(set1, set2):
  return mean_squared_error(set1, set2)

def main():
  # Assignment 02.2
  # a) b)
  for i in range(1,30 + 1):
    print "k = " + str(i)
    hilbertMatrix = generateHilbertMatrix(i)
    calculateRankAndConditionNumbers(hilbertMatrix)

  # c)
  for j in [1,2,3,5,10,15,20,30,50,100]:
    print "x_k for k = " + str(j) + ":"
    solveEquations(generateHilbertMatrix(j), np.ones(j))

  # d) Only to some extend (see subtask e)

  # e)
  # The check result should return only 0s, but since the matrix gets more
  # inaccurate the bigger it gets due to the floating point calculation
  # 1/x the results are already (a little bit) wrong for k = 3


  # Assignment 02.3
  # a)
  data = readFileData()
  data = data[:, ~np.isnan(data).any(axis=0)]

  # numCols is -1 because the last column ocean_proximity is only strings
  numCols = data.shape[1] - 1
  numRows = data.shape[0]

  # b)
  for i in range(0, numCols):
    findMinMaxMean(data[:,i].reshape(numRows,1))

  # c)
  # I would say they all have a normal distribution, but not the same ones
  columnNames = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "median_house_value", "ocean_proximity"]

  for i in range(0,numCols):
    plt.figure()
    plt.title("Histogram for data column " + columnNames[i])
    plt.hist(data[:,i])

  # d)
  plt.figure()
  plt.title("Geographical map of houses")
  plt.scatter(data[:,0], data[:,1], s=2, alpha=0.1, c=data[:,7], cmap=cm.bwr)

  # e)
  labels = range(0,numRows)
  values = data

  X_train, X_test, y_train, y_test = train_test_split(values, labels, test_size=0.20, random_state=42)

  findMinMaxMean(X_train)
  findMinMaxMean(X_test)

  plt.figure()
  plt.title("Train set histogram")
  plt.hist(X_train)

  plt.figure()
  plt.title("Test set histogram")
  plt.hist(X_test)
  
  # --> the test and training set distribution match

  # plots all figures
  #plt.show()


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
  print loss

  # b)
  distance = euclideanDist(data[:,0], data[:,1])
  # TODO distance ist viel zu groß?!
  # vielleicht sollte für kNN die Funktion euclideanDistForTwoPoints()
  # verwendet werden..
  print 'Distance: ' + repr(distance)

  # c)
  # implement kNN
  # siehe https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/


  # d)
  # predict housing values for different values of k
  # what are training and test errors?

# execute main
if __name__ == "__main__":
  main()
