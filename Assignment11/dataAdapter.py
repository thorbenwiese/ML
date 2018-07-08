# -*- coding: utf-8 -*-

import csv

def readCSV(iteration):
    ids = []
    vectors = []
    labels = []
    with open('resultTRAIN{}.csv'.format(iteration), 'rU') as data:
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

def readTestCSV(iteration):
    test_ids = []
    test_vectors = []
    test_labels = []
    with open('resultTEST{}.csv'.format(iteration), 'rU') as data:
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