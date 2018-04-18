import numpy as np

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

def main():
# Assignment 02.2
  for i in range(1,30 + 1):
    print "k = " + str(i)
    hilbertMatrix = generateHilbertMatrix(i)
    calculateRankAndConditionNumbers(hilbertMatrix)

  for j in [1,2,3,5,10,15,20,30,50,100]:
    print "x_k for k = " + str(j) + ":"
    solveEquations(generateHilbertMatrix(j), np.ones(j))

# The check result should return only 0s, but since the matrix gets more
# inaccurate the bigger it gets due to the floating point calculation
# 1/x the results are already wrong for k = 3


# Assignment 02.3
  data = readFileData()
  print data[:,0]
  print len(data)
  for i in range(0, 9):
    print "i: " + str(i)
  for i in range(0, 9):
    findMinMaxMean(data[:,i].reshape(len(data),1))




if __name__ == "__main__":
  main()
