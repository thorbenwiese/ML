# -*- coding: utf-8 -*-

import csv
import matplotlib.pyplot as plt
import numpy as np

def readCSV():
    ids = []
    vectors = []
    labels = []
    with open('resultTRAIN.csv', 'rU') as data:
        reader = csv.reader(data)
        count = 0
        for row in reader:
            if count != 0:
              ids.append(row[0])
              s = row[1].replace('[','').replace(']','').split(', ')
              tmp = [float(i) for i in s]
              vectors.append(tmp)
              labels.append(float(row[2]))
            count = 42
    return ids, vectors, labels

# PLOTTING IMAGES -> TODO: Sind die sinnvoll?

#plt.figure()
#[plt.scatter(v, range(len(v)), s=1) for v in vectors]

#plt.figure()
#[plt.plot(v) for v in vectors]

#plt.show()

def readTestCSV():
    test_ids = []
    test_vectors = []
    test_labels = []
    with open('resultTEST.csv', 'rU') as data:
        reader = csv.reader(data)
        count = 0
        for row in reader:
            if count != 0:
              test_ids.append(row[0])
              s = row[1].replace('[','').replace(']','').split(', ')
              tmp = [float(i) for i in s]
              test_vectors.append(tmp)
              test_labels.append(float(row[2]))
            count = 42
    return test_ids, test_vectors, test_labels


ids, vectors, labels = readCSV()
test_ids, test_vectors, test_labels = readTestCSV()

# PREDICTION WITH KNN
from sklearn.neighbors import KNeighborsClassifier

k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(vectors, labels)

# TODO this needs to be test data with the correct shape...
#tmp = labels[:23]
#acc = model.score(test_vectors, tmp)
acc = model.score(test_vectors, test_labels)
print ('KNN Accuracy:', acc)

print (np.shape(vectors))
print (np.shape(labels))
print (np.shape(test_vectors))
#print (np.shape(tmp))
print (np.shape(test_labels))

