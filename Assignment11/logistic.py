# -*- coding: utf-8 -*-
from sklearn import  linear_model
import time
import numpy as np
import dataAdapter
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt


result = []
# iterations = [200, 2000]  # , 20000]
# Cs = [0.01, 0.1, 0.5, 1, 5, 10, 50]
Cs = [0.1, 0.5, 1, 10]
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
#gammas = [0.1, 1, 10]  # , 100]
fignum = 1

def test_solver(iteration, input_train, target_train, input_test, target_test, title):
    result.append([])
    for kern in solver:
        errors = []
        test_error = []
        for c in Cs:
            start = time.time()
            logistic = linear_model.LogisticRegression(C=c, class_weight=None, dual=False,
                                                       fit_intercept=True, intercept_scaling=1, max_iter=100,
                                                       multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
                                                       solver=kern, tol=0.0001, verbose=0, warm_start=False)
            model = logistic.fit(input_train, target_train)
            # error = 1- model.score(input_train, target_train)
            error = zero_one_loss(target_train, logistic.predict(input_train))
            errors.append(error)
            print('iteration: ', iteration, 'c: ', c, ' kernel: ', kern,  " train:", error)
            prediction = logistic.predict(input_test)
            t_error = zero_one_loss(target_test, prediction)
            test_error.append(t_error)
            print('iteration: ', iteration, 'c: ', c,  ' kernel: ', kern, " test:", t_error)
            end = time.time()
            duration = end - start
            result[-1].append( ((kern,  c),(1 - error, 1- t_error, duration) ) )


def logistic_regression(iterations):

    global result
    result = []

    for iteration in iterations:
        ids, vectors, labels = dataAdapter.readCSV(iteration)
        test_ids, test_vectors, test_labels = dataAdapter.readTestCSV(iteration)

        print('ITERATION {}'.format(iteration))
        test_solver(iteration, vectors, labels, test_vectors, test_labels, iteration)
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

logistic_regression([200])
#logistic_regression([2000])
