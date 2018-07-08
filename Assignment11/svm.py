import dataAdapter
import matplotlib.pyplot as plt
import time
import sys
from sklearn.svm import SVC
from sklearn.metrics import zero_one_loss
import numpy as np

#f = open('svm_result','w')
#sys.stdout = f

result = []
# iterations = [200, 2000]  # , 20000]
# Cs = [0.01, 0.1, 0.5, 1, 5, 10, 50]
Cs = [0.1, 1, 10]
kernels = ['linear', 'poly', 'rbf', 'sigmoid']  # , 'precomputed']
gammas = [0.1, 1, 10]  # , 100]
fignum = 1

def test_kernels(iteration, input_train, target_train, input_test, target_test, plottrain, title):
    result.append([])
    f, axarr = plt.subplots(2, 2)
    counter = 0
    counter2 = 0
    for gamma in gammas:
        for kern in kernels:
            errors = []
            test_error = []
            for c in Cs:
                start = time.time()
                clf = SVC(C=c, kernel=kern, degree=3, gamma=gamma, coef0=0.0, shrinking=True,
                          probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
                          max_iter=-1, decision_function_shape='ovr', random_state=None)
                model = clf.fit(input_train, target_train)
                # error = 1- model.score(input_train, target_train)
                error = zero_one_loss(target_train, clf.predict(input_train))
                errors.append(error)
                print('iteration: ', iteration, 'c: ', c, 'gamma: ', gamma, ' kernel: ', kern,  " train:", error)
                prediction = clf.predict(input_test)
                t_error = zero_one_loss(target_test, prediction)
                test_error.append(t_error)
                print('iteration: ', iteration, 'c: ', c, 'gamma: ', gamma, ' kernel: ', kern, " test:", t_error)
                end = time.time()
                duration = end - start
                result[-1].append( ((kern, gamma, c),(1 - error, 1- t_error, duration) ) )
            if plottrain:
                axarr[int(counter / 2), counter2 % 2].plot(range(len(Cs)), errors, label=kern + ' train')
            axarr[int(counter / 2), counter2 % 2].plot(range(len(Cs)), test_error, label=kern + ' test')
        axarr[int(counter / 2), counter2 % 2].set_title('gamma=' + str(gamma))
        counter += 1
        counter2 += 1
    plt.legend(bbox_to_anchor=(1.05, 1), loc=3, borderaxespad=0.)
    plt.setp(axarr, xticks=range(len(Cs)), xticklabels=Cs, xlabel='C', ylabel='error')
    # plt.title(title)
    f.canvas.set_window_title(title)
    plt.xlabel('C')
    plt.ylabel('error')
    global fignum
    fignum += 1


def svm(iterations):

    global result
    result = []

    for iteration in iterations:
        ids, vectors, labels = dataAdapter.readCSV(iteration)
        test_ids, test_vectors, test_labels = dataAdapter.readTestCSV(iteration)

        print('ITERATION {}'.format(iteration))
        test_kernels(iteration, vectors, labels, test_vectors, test_labels, True, iteration)
        print('\n')
        print(30 * '=')
        print('\n')

    print(result)

    maxtime = -1
    for x in result:
        maxtime = max([maxtime, max([l[1][2] for l in x])])
    print('maxtime: ', maxtime)

    x1 = result[0]
    N = sum(map(len, result))
    y_labels = [x[0] for x in x1]
    train_errors = [x[1][0] for x in x1]
    test_errors = [x[1][1] for x in x1]
    durations = [x[1][2] / maxtime for x in x1]

    ind = np.arange(N)  # the x locations for the groups
    width = 0.25  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.barh(ind, train_errors[:N], width, color='r')
    rects2 = ax.barh(ind + width, test_errors[:N], width, color='y')
    rects3 = ax.barh(ind + 2 * width, durations[:N], width, color='b')

    y_pos = np.arange(N)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)

    ax.legend((rects1[0], rects2[0], rects3[0]), ('train', 'test', 'time'))
    print(result)
    plt.show()

#svm([200])
svm([2000])







'''
 for x in x1:
        counter +=1
        plt.bar(X + 0.00, x[1][0], color='b', width=0.25)
        plt.bar(X + 0.25, x[1][1], color='g', width=0.25)
        plt.bar(X + 0.50, x[1][2] / maxtime, color='r', width=0.25)
'''
