import numpy as np

# Assignment 02.2
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





def main():
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

if __name__ == "__main__":
  main()
