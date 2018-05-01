# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio


'''
Read the vaccination.csv data into your Python workspace. Determine the numbers 
of boys/girls, age groups and olderSiblings. Visualize these numbers with bar 
plots.
'''
def readCsvFile():
  '''
  gender
  age
  height
  weight
  residence
  olderSiblings
  knowsToRideABike
  vacX
  diseaseX
  diseaseY
  diseaseZ
  '''
  data = np.genfromtxt('vaccination.csv', delimiter=',')
  return np.transpose(data[1:])

def determineGroups(data):
  gender = data[0]
  numBoys = len(gender[gender == 1])
  numGirls = len(gender[gender == 0])
  print '# boys: ', numBoys
  print '# girls: ', numGirls
  plt.figure()
  plt.subplot(1,3,1)
  plt.title('Number of boys and girls')
  plt.bar([1,2], [numBoys, numGirls])
  plt.xticks([1,2], ('Boys', 'Girls'))
  plt.xlabel('Gender')
  plt.ylabel('Number of children')

  age = data[1]
  numAge1 = len(age[age == 1])
  numAge2 = len(age[age == 2])
  numAge3 = len(age[age == 3])
  numAge4 = len(age[age == 4])
  numAge5 = len(age[age == 5])
  print '# children between 0-2 years: ', numAge1
  print '# children between 3-6 years: ', numAge2
  print '# children between 7-10 years: ', numAge3
  print '# children between 11-13 years: ', numAge4
  print '# children between 14-17 years: ', numAge5
  plt.subplot(1,3,2)
  plt.title('Number of children in different age groups')
  plt.bar([1,2,3,4,5], [numAge1, numAge2, numAge3, numAge4, numAge5])
  plt.xticks([1,2,3,4,5], ('0-2', '3-6', '7-10', '11-13', '14-17'))
  plt.xlabel('Age groups')
  plt.ylabel('Number of children')

  olderSiblings = data[5]
  siblings0 = len(age[olderSiblings == 0])
  siblings1 = len(age[olderSiblings == 1])
  siblings2 = len(age[olderSiblings == 2])
  siblings3 = len(age[olderSiblings == 3])
  siblings4 = len(age[olderSiblings == 4])
  print '# children without older siblings: ', siblings0
  print '# children with 1 older siblings: ', siblings1
  print '# children with 2 older siblings: ', siblings2
  print '# children with 3 older siblings: ', siblings3
  print '# children with 4 older siblings: ', siblings4
  plt.subplot(1,3,3)
  plt.title('Number of children with older siblings')
  plt.bar([1,2,3,4,5], [siblings0, siblings1, siblings2, siblings3, siblings4])
  plt.xticks([1,2,3,4,5], ('0', '1', '2', '3', '4'))
  plt.xlabel('Number of older siblings')
  plt.ylabel('Number of children')
  print ''

'''
4.1b)
'''
def calculateMarginalProbabilities(data):
  vaccination = data[7]
  allRows = len(vaccination)
  hasVaccAgainstX = (1.0 * len(vaccination[vaccination == 1])) / allRows
  print 'Probability to have a vaccination against disease X: ', hasVaccAgainstX

  residence = data[4]
  allRows = len(residence)
  countrySide = residence[residence == 1]
  livesOnCountrySide = (1.0 * len(residence[residence == 1])) / allRows
  print 'Probability to live on the country side: ', livesOnCountrySide

  olderSiblings = data[5]
  allRows = len(olderSiblings)
  hasOlderSiblings = (1.0 * len(olderSiblings[olderSiblings > 0])) / allRows
  print 'Probability to have at least one older sibling: ', hasOlderSiblings
  print ''

'''
4.1.c)
'''
def preprocessing(data):
  height = data[2]
  allRows = len(height)
  tallerThanOneMeter = (1.0 * len(height[height > 100])) / allRows
  print 'Probability to be taller than 1 meter: ', tallerThanOneMeter

  weight = data[3]
  allRows = len(weight)
  heavierThan40kg = (1.0 * len(weight[weight > 40])) / allRows
  print 'Probability to be heavier than 40 kg: ', heavierThan40kg

  diseaseY = data[9]
  diseaseZ = data[10]
  allRows = len(diseaseY)
  posYZ = diseaseY[(diseaseY == 1) & (diseaseZ == 1)]
  posY = diseaseY[(diseaseY == 1) & (diseaseZ == 0)]
  posZ = diseaseZ[(diseaseY == 0) & (diseaseZ == 1)]
  hasHadYOrZ = (1.0 * (len(posY) + len(posZ) + len(posYZ))) / allRows
  print 'Probability to have had diseaseY and/or diseaseZ: ', hasHadYOrZ
  print ''

'''
4.1.d)
'''
def conditionalProbabilities(data):
  plt.figure()
  plt.title('Conditional Probabilities')

  diseaseX = data[8]
  vacX = data[7]
  numDiseaseXNotVacX = len(diseaseX[(diseaseX == 1) & (vacX == 0)])
  numNotVacX = len(vacX[vacX == 0])
  proba = (1.0 * numDiseaseXNotVacX) / numNotVacX
  print 'P(diseaseX = 1 | vacX = 0) = ', proba
  numDiseaseXVacX = len(diseaseX[(diseaseX == 1) & (vacX == 1)])
  numVacX = len(vacX[vacX == 1])
  proba = (1.0 * numDiseaseXVacX) / numVacX
  print 'P(diseaseX = 1 | vacX = 1) = ', proba

  proba = (1.0 * numDiseaseXVacX) / len(diseaseX[diseaseX == 0])
  print 'P(vacX = 1 | diseaseX = 0) = ', proba
  proba = (1.0 * numDiseaseXVacX) / len(diseaseX[diseaseX == 1])
  print 'P(vacX = 1 | diseaseX = 1) = ', proba

  diseaseY = data[9]
  age = data[1]
  numDiseaseYAge1 = len(diseaseY[(diseaseY == 1) & (age == 1)])
  numAge1 = len(age[age == 1])
  pDisAge1 = (1.0 * numDiseaseYAge1) / numAge1
  print 'P(diseaseY = 1 | age = 1) = ', pDisAge1
  numDiseaseYAge2 = len(diseaseY[(diseaseY == 1) & (age == 2)])
  numAge2 = len(age[age == 2])
  pDisAge2 = (1.0 * numDiseaseYAge2) / numAge2
  print 'P(diseaseY = 1 | age = 2) = ', pDisAge2
  numDiseaseYAge3 = len(diseaseY[(diseaseY == 1) & (age == 3)])
  numAge3 = len(age[age == 3])
  pDisAge3 = (1.0 * numDiseaseYAge3) / numAge3
  print 'P(diseaseY = 1 | age = 3) = ', pDisAge3
  numDiseaseYAge4 = len(diseaseY[(diseaseY == 1) & (age == 4)])
  numAge4 = len(age[age == 4])
  pDisAge4 = (1.0 * numDiseaseYAge4) / numAge4
  print 'P(diseaseY = 1 | age = 4) = ', pDisAge4
  plt.subplot(2,1,1)
  plt.plot([1,2,3,4], [pDisAge1, pDisAge2, pDisAge3, pDisAge4])
  plt.xlabel('Age')
  plt.ylabel('P(disaseY|age)')

  numVacXAge1 = len(vacX[(vacX == 1) & (age == 1)])
  pVacAge1 = (1.0 * numVacXAge1) / numAge1
  print 'P(vacX = 1 | age = 1) = ', pVacAge1
  numVacXAge2 = len(vacX[(vacX == 1) & (age == 2)])
  pVacAge2 = (1.0 * numVacXAge2) / numAge2
  print 'P(vacX = 1 | age = 2) = ', pVacAge2
  numVacXAge3 = len(vacX[(vacX == 1) & (age == 3)])
  pVacAge3 = (1.0 * numVacXAge3) / numAge3
  print 'P(vacX = 1 | age = 3) = ', pVacAge3
  numVacXAge4 = len(vacX[(vacX == 1) & (age == 4)])
  pVacAge4 = (1.0 * numVacXAge4) / numAge4
  print 'P(vacX = 1 | age = 4) = ', pVacAge4
  plt.subplot(2,1,2)
  plt.plot([1,2,3,4], [pVacAge1, pVacAge2, pVacAge3, pVacAge4])
  plt.xlabel('Age')
  plt.ylabel('P(vacX|age)')

  knowsToRideABike = data[6]
  numKtrabNotVacX = len(knowsToRideABike[(knowsToRideABike == 1) & (vacX == 0)])
  proba = (1.0 * numKtrabNotVacX) / numNotVacX
  print 'P(knowsToRideABike = 1 | vacX = 0) = ', proba
  numKtrabVacX = len(knowsToRideABike[(knowsToRideABike == 1) & (vacX == 1)])
  proba = (1.0 * numKtrabVacX) / numVacX
  print 'P(knowsToRideABike = 1 | vacX = 1) = ', proba

  '''
  Interpretation:
  The probability of having had diseaseX increases by age.
  The probability of being vaccinated against diseaseX increases by age.
  --> The vaccination does not protect the children against the disease.
  '''

def main():
  data = readCsvFile()
  determineGroups(data)
  calculateMarginalProbabilities(data)
  preprocessing(data)
  conditionalProbabilities(data)

  plt.show()


if __name__ == '__main__':
  main()
