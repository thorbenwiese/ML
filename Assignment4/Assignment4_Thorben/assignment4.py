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
  print '-----------------'
  print '| Aufgabe 4.1.a |'
  print '-----------------\n'
  gender = data[0]
  numBoys = len(gender[gender == 1])
  numGirls = len(gender[gender == 0])
  print '# boys: ', numBoys
  print '# girls: ', numGirls
  plt.figure(figsize=(20,5))
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
  print '-----------------'
  print '| Aufgabe 4.1.b |'
  print '-----------------\n'
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
  print '-----------------'
  print '| Aufgabe 4.1.c |'
  print '-----------------\n'
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
4.1.e)
'''
def conditionalProbabilities(data):
  print '-----------------'
  print '| Aufgabe 4.1.d |'
  print '-----------------\n'
  plt.figure(figsize=(7,7))
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
  plt.xticks([1,2,3,4])

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
  plt.xticks([1,2,3,4])

  knowsToRideABike = data[6]
  numKtrabNotVacX = len(knowsToRideABike[(knowsToRideABike == 1) & (vacX == 0)])
  proba = (1.0 * numKtrabNotVacX) / numNotVacX
  print 'P(knowsToRideABike = 1 | vacX = 0) = ', proba
  numKtrabVacX = len(knowsToRideABike[(knowsToRideABike == 1) & (vacX == 1)])
  proba = (1.0 * numKtrabVacX) / numVacX
  print 'P(knowsToRideABike = 1 | vacX = 1) = ', proba
  print ''

  '''
  Interpretation:
  The probability of having had diseaseX increases by age.
  The probability of being vaccinated against diseaseX increases by age.
  --> The vaccination does not protect the children against the disease.
  --> Maybe the absolute values should be taken into account instead of only
      relative values (Simpson's paradox)
  '''

  # 4.1.e
  print '-----------------'
  print '| Aufgabe 4.1.e |'
  print '-----------------\n'
  diseaseZ = data[10]
  numDiseaseYZvacX = len(diseaseY[((diseaseY == 1) | (diseaseZ == 1)) & (vacX == 1)])
  proba = (1.0 * numDiseaseYZvacX) / numVacX
  print 'P(diseaseYZ = 1 | vacX = 1) = ', proba
  numDiseaseYZNotVacX = len(diseaseY[((diseaseY == 1) | (diseaseZ == 1)) & (vacX == 0)])
  proba = (1.0 * numDiseaseYZNotVacX) / numNotVacX
  print 'P(diseaseYZ = 1 | vacX = 0) = ', proba

  numDiseaseXNotVacX = len(diseaseX[(diseaseX == 1) & (vacX == 0)])
  numNotVacX = len(vacX[vacX == 0])
  proba = (1.0 * numDiseaseXNotVacX) / numNotVacX
  print 'P(diseaseX = 1 | vacX = 0) = ', proba
  numDiseaseXVacX = len(diseaseX[(diseaseX == 1) & (vacX == 1)])
  numVacX = len(vacX[vacX == 1])
  proba = (1.0 * numDiseaseXVacX) / numVacX
  print 'P(diseaseX = 1 | vacX = 1) = ', proba
  print ''

  numDiseaseYZvacXage1 = len(diseaseY[((diseaseY == 1) | (diseaseZ == 1)) & (vacX == 1) & (age == 1)])
  p11 = (1.0 * numDiseaseYZvacXage1) / numVacX
  print 'P(diseaseYZ = 1 | vacX = 1, age = 1) = ', p11
  numDiseaseYZNotVacXage1 = len(diseaseY[((diseaseY == 1) | (diseaseZ == 1)) & (vacX == 0) & (age == 1)])
  p12 = (1.0 * numDiseaseYZNotVacXage1) / numNotVacX
  print 'P(diseaseYZ = 1 | vacX = 0, age = 1) = ', p12

  numDiseaseYZvacXage2 = len(diseaseY[((diseaseY == 1) | (diseaseZ == 1)) & (vacX == 1) & (age == 2)])
  p21 = (1.0 * numDiseaseYZvacXage2) / numVacX
  print 'P(diseaseYZ = 1 | vacX = 1, age = 2) = ', p21
  numDiseaseYZNotVacXage2 = len(diseaseY[((diseaseY == 1) | (diseaseZ == 1)) & (vacX == 0) & (age == 2)])
  p22 = (1.0 * numDiseaseYZNotVacXage2) / numNotVacX
  print 'P(diseaseYZ = 1 | vacX = 0, age = 2) = ', p22

  numDiseaseYZvacXage3 = len(diseaseY[((diseaseY == 1) | (diseaseZ == 1)) & (vacX == 1) & (age == 3)])
  p31 = (1.0 * numDiseaseYZvacXage3) / numVacX
  print 'P(diseaseYZ = 1 | vacX = 1, age = 3) = ', p31
  numDiseaseYZNotVacXage3 = len(diseaseY[((diseaseY == 1) | (diseaseZ == 1)) & (vacX == 0) & (age == 3)])
  p32 = (1.0 * numDiseaseYZNotVacXage3) / numNotVacX
  print 'P(diseaseYZ = 1 | vacX = 0, age = 3) = ', p32

  numDiseaseYZvacXage4 = len(diseaseY[((diseaseY == 1) | (diseaseZ == 1)) & (vacX == 1) & (age == 4)])
  p41 = (1.0 * numDiseaseYZvacXage4) / numVacX
  print 'P(diseaseYZ = 1 | vacX = 1, age = 4) = ', p41
  numDiseaseYZNotVacXage4 = len(diseaseY[((diseaseY == 1) | (diseaseZ == 1)) & (vacX == 0) & (age == 4)])
  p42 = (1.0 * numDiseaseYZNotVacXage4) / numNotVacX
  print 'P(diseaseYZ = 1 | vacX = 0, age = 4) = ', p42
  print ''

  '''
  How sure are you that your estimates for P (diseaseY Z | vacX = 0/1, 
  age = 1/2/3/4) are accurate? What does this depend on?

  --> We cannot be sure that the results are accurate, because they only depend
      on the relative occurences in the data and do not consider the absolute
      occurences. If the children are equally distributed by age then the
      results will be accurate.
  '''

  plt.figure()
  plt.title('DiseaseYZ, age=1/2/3/4 with and without vacX')
  plt.plot([1,2,3,4], [p11,p21,p31,p41], label='vacX = 1')
  plt.plot([1,2,3,4], [p12,p22,p32,p42], label='vacX = 0')
  plt.xticks([1,2,3,4])
  plt.xlabel('Age')
  plt.ylabel('P(diseaseYZ|vacX=0/1|age)')
  plt.legend()

  '''
  What do you conclude from your plot?

  --> The probability to get diseaseY or diseaseZ if the child was vaccinated
      against diseaseX increases by age.
      The probability to get diseaseY or diseaseZ if the child was not
      vaccinated against diseaseX is low for age 1 and 4 and higher for age 
      2 and 3.
      We conclude that getting a vaccination against diseaseX increases the
      chance to get diseaseY or diseaseZ and that the vaccination against
      diseaseX is more effective at age 1 and 4.
  '''

def aufgabe2():
  '''
  4.2.a
  '''
  data = sio.loadmat('reg1d.mat')
  X_train_ori = data['X_train']
  X_test = data['X_test']
  Y_train_ori = data['Y_train']
  Y_test = data['Y_test']

  r = range(len(X_train_ori))

  plt.figure()
  plt.title('Train and Test Data')
  plt.scatter(X_train_ori, Y_train_ori, s=3, label='Train Data')
  plt.scatter(X_test, Y_test, s=3, label='Test Data')

  '''
  4.2.b
  '''
  print '-----------------'
  print '| Aufgabe 4.2.b |'
  print '-----------------\n'

  X_train = np.vander(X_train_ori.flatten(), 2, False)
  Y_train = np.vander(Y_train_ori.flatten(), 2, False)

  coefs1 = least_squares(X_train, Y_train)
  print 'Coefs1: ', coefs1, '\n'

  func1 = np.poly1d(coefs1)
  pred1 = func1(r)

  plt.plot(pred1, label='Pred 1 dim')
  plt.xlim(0,1)
  plt.ylim(0,10)

  '''
  4.2.d
  '''
  print '-----------------'
  print '| Aufgabe 4.2.d |'
  print '-----------------\n'

  X_train = np.vander(X_train_ori.flatten(), 4, False)
  Y_train = np.vander(Y_train_ori.flatten(), 4, False)

  coefs3 = least_squares(X_train, Y_train)
  print 'Coefs3: ', coefs3, '\n'

  func3 = np.poly1d(coefs3)
  pred3 = func3(r)
  plt.plot(pred3, label='Pred 3 dim')

  # --
  '''
  for i in [2,3]:
    #[1,2,3,4,5,15]:
    pf = np.polyfit(X_train_ori.flatten(), Y_train_ori.flatten(), i)
    p = np.poly1d(pf)
    pr = p(r)
    print p(r)
    plt.plot(r, pr, label='Polyfit ' + str(i))
  '''
  # --

  plt.legend()

  '''
  4.2.c
  '''
  err1 = lossL2(Y_test, pred1)
  print 'Err1: ', err1, '\n'

  err3 = lossL2(Y_test, pred3)
  print 'Err3: ', err3, '\n'

  '''
  4.2.e
  '''
  print '-----------------'
  print '| Aufgabe 4.2.e |'
  print '-----------------\n'

  X_train_ori = np.vstack((X_train_ori, 1.05))
  Y_train_ori = np.vstack((Y_train_ori, -10))

  plt.figure()
  plt.title('Train and Test Data with Outlier')
  plt.scatter(X_train_ori, Y_train_ori, s=3, label='Train Data')
  plt.scatter(X_test, Y_test, s=3, label='Test Data')

  r = range(len(X_train_ori))

  X_train = np.vander(X_train_ori.flatten(), 2, False)
  Y_train = np.vander(Y_train_ori.flatten(), 2, False)

  coefs1 = least_squares(X_train, Y_train)
  print 'Coefs1: ', coefs1, '\n'

  func1 = np.poly1d(coefs1)
  pred1 = func1(r)

  plt.plot(pred1, label='Pred 1 dim')
  plt.xlim(0,1)
  plt.ylim(0,10)

  X_train = np.vander(X_train.flatten(), 4, False)
  Y_train = np.vander(Y_train.flatten(), 4, False)

  coefs3 = least_squares(X_train, Y_train)
  print 'Coefs3: ', coefs3, '\n'

  func3 = np.poly1d(coefs3[::-1])
  pred3 = func3(r)
  plt.plot(pred3, label='Pred 3 dim')


  plt.legend()

  err1 = lossL2(Y_test, pred1)
  print 'Err1: ', err1, '\n'

  err3 = lossL2(Y_test, pred3)
  print 'Err3: ', err3

'''
4.2.b
'''
def least_squares(X, Y):
  # solve w = (X.T X)^-1 X.T Y
  # This is too inaccurate
  #return np.linalg.solve(X.T.dot(X), X.T.dot(Y))
  return np.linalg.lstsq(X, Y)[0].flatten()

'''
4.2.c
'''
def lossL2(Y, Y_pred):
  return np.mean(np.power(np.array(Y)-np.array(Y_pred), 2))


def main():
  data = readCsvFile()
  determineGroups(data)
  calculateMarginalProbabilities(data)
  preprocessing(data)
  conditionalProbabilities(data)

  aufgabe2()

  plt.show()


if __name__ == '__main__':
  main()
