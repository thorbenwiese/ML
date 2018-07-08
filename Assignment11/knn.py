# -*- coding: utf-8 -*-
import dataAdapter
import matplotlib.pyplot as plt
import time
import sys
from sklearn.neighbors import KNeighborsClassifier

f = open('knn_result','w')
sys.stdout = f

plotfigures = False
iterations = [200, 200, 20000]
ks = [2,3,4,5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 75, 100, 125, 150, 200, 500, 750, 1000] # > 150 doesn't work for iteration = 200

result = {'acc':[], 'time':[]}

for iteration in iterations:
    #ids, vectors, labels = dataAdapter.readCSV('') # -> '' is default if filename is resultTRAIN.csv instead resultTRAIN<iteration>.csv
    #test_ids, test_vectors, test_labels = dataAdapter.readTestCSV('')
    result['acc'].append([])
    result['time'].append([])

    ids, vectors, labels = dataAdapter.readCSV( iteration)
    test_ids, test_vectors, test_labels = dataAdapter.readTestCSV(iteration)

    '''
    # PLOTTING IMAGES -> TODO: Sind die sinnvoll?
    if plotfigures:
        plt.figure()
        [plt.scatter(v, range(len(v)), s=1) for v in vectors]

        plt.figure()
        [plt.plot(v) for v in vectors]
    '''


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
        duration = end - start
        result['acc'][-1].append(acc)
        result['time'][-1].append(duration)
        print ( 'Configuration: size = {}, k= {}'.format(iteration, k))
        print ('KNN Accuracy:', acc)
        print ('Time consumption: ', duration)
        print ( 30*'-')
    '''
    if plotfigures:
        plt.show()
    '''
print(result)

fig, ax = plt.subplots(2, sharex=True)
for a in range(len(result['acc'])):
    ax[0].plot(result['acc'][a], label = 'iteration = {}'.format(iterations[a]))
    ax[1].plot(result['time'][a], label='iteration = {}'.format(iterations[a]))
plt.xlabel('k')
ax[0].set_title('acc')
ax[1].set_title('time')
ax[0].set_xticklabels(ks)
ax[1].set_xticklabels(ks)
legend = plt.legend()
fig.savefig('knn.png')   # save the figure to file
plt.close(fig)

