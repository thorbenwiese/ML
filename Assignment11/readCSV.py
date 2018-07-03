# -*- coding: utf-8 -*-

import csv
import matplotlib.pyplot as plt

def readCSV():
    ids = []
    vectors = []
    with open('result.csv', 'rU') as data:
        reader = csv.reader(data)
        count = 0
        for row in reader:
            if count != 0:
              ids.append(row[0])
              s = row[1].replace('[','').replace(']','').split(', ')
              tmp = [float(i) for i in s]
              vectors.append(tmp)
            count = 42
    return ids, vectors

ids, vectors = readCSV()

plt.figure()
[plt.scatter(v, range(len(v)), s=1) for v in vectors]

plt.figure()
[plt.plot(v) for v in vectors]

plt.show()
