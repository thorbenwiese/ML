# -*- coding: utf-8 -*-
import dataAdapter
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import KNeighborsClassifier

plotfigures = False
iterations = [200, 2000]#, 20000]
ks = [2,3,4,5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 75, 100, 125, 150, 200, 500, 750, 1000] # > 150 doesn't work for iteration = 200

for iteration in iterations:
    #ids, vectors, labels = dataAdapter.readCSV('') # -> '' is default if filename is resultTRAIN.csv instead resultTRAIN<iteration>.csv
    #test_ids, test_vectors, test_labels = dataAdapter.readTestCSV('')

    ids, vectors, labels = dataAdapter.readCSV( iteration)
    test_ids, test_vectors, test_labels = dataAdapter.readTestCSV(iteration)

    # PLOTTING IMAGES -> TODO: Sind die sinnvoll?
    if plotfigures:
        plt.figure()
        [plt.scatter(v, range(len(v)), s=1) for v in vectors]

        plt.figure()
        [plt.plot(v) for v in vectors]


    # PREDICTION WITH KNN

    for k in ks:
        if iteration == 200 and k > 149:
            break
        start = time.time()
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(vectors, labels)

        # TODO this needs to be test data with the correct shape...
        #tmp = labels[:23]
        #acc = model.score(test_vectors, tmp)
        acc = model.score(test_vectors, test_labels)
        end = time.time()
        print ( 'Configuration: size = {}, k= {}'.format(iteration, k))
        print ('KNN Accuracy:', acc)
        print ('Time consumption: ', end - start)
        print ( 30*'-')

    if plotfigures:
        plt.show()

