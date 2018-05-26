import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import zero_one_loss
from matplotlib.pyplot import plot

######## 1b ###############
'''
How do you configure and train the SVM
TODO SVC Konfiguration allgemein
For training we use fit(X, y[, sample_weight]), where X is a list of training vectors and y contains 
the related target values. There is the possibility to weight samples. 


How do you classify new test points?
predict(X) performs a classification on samples in X

How do you access the support vectors after training?
The support vectors are stored as attribute clf.support_vectors_
e.g. print(clf.support_vectors_)


How do you configure the SVM for multi-class classification problems?
The multiclass support is handled according to a one-vs-one scheme. 
We can use OneVsOneClassifier, which constructs one classifier per pair of classes. 
E.g. OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
Alternatively we can use the one-vs-all-strategy.  For each classifier, the class 
is fitted against all the other classes. Advantage of this ate the computational 
efficiency and interpretability. This is the most commonly used strategy and is a fair default choice.
E.g. OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
In LinearSVC multi_class="ovr" trains n_classes one-vs-rest classifiers, while "crammer_singer" optimizes a joint objective over all classes.
(Details siehe http://scikit-learn.org/stable/modules/multiclass.html)


    
######## 2a ###############
'''
data = sio.loadmat('cancer-data.mat')
input_train = data['cancerInput_train']
target_train = data['cancerTarget_train'].flatten()
input_test = data['cancerInput_test']
target_test = data['cancerTarget_test'].flatten()

Cs = [0.01, 0.1, 0.5, 1, 5, 10, 50]
errors =[]
test_error = []
for c in Cs:
    clf = LinearSVC(C=c, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
         verbose=0)
    model = clf.fit(input_train, target_train)
    error = 1- model.score(input_train, target_train)
    errors.append(error)
    prediction = clf.predict(input_test)
    t_error = zero_one_loss(target_test, prediction)
    test_error.append(t_error)
    print(c, " train:", error) #TODO richtig für train error?
    print(c, " test:", t_error)
plt.plot(range(len(Cs)), errors, label ='train_error')
plt.plot(range(len(Cs)), test_error, label='test_error')
legend = plt.legend(loc='upper center', shadow=True)
plt.ylim(0, 0.05)
plt.xticks(np.arange(7), Cs)
plt.xlabel('C')
plt.ylabel('error')
plt.title('2a')
plt.show()
    #TODO 0 1 loss?
    #prediction = clf.predict(input_test)
    #print(c, ":", zero_one_loss(target_test, prediction))
    
'''
What is the effect of choosing a large C on the training error?
Choosing a large C results in a decreasing training error.

Does this effect coincide with what you are expecting?
C is the penalty parameter of the error term. So a increasing C tells the SVM optimization 
how much you want to avoid misclassifying each training example.
TODO expected
'''


######## 2b ###############
def test_kernels(t, t_label, title):
    for kern in ['linear', 'poly', 'rbf', 'sigmoid']:#, 'precomputed']:
        errors = []
        test_error = []
        for c in Cs:
            clf = SVC(C=c, kernel=kern, degree=3, gamma='auto', coef0=0.0, shrinking=True, 
                      probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, 
                      max_iter=-1, decision_function_shape='ovr', random_state=None)
            model = clf.fit(t, t_label)
            error = 1- model.score(t, t_label)
            errors.append(error)
            print(kern, c, "train:", error)
            prediction = clf.predict(input_test)
            t_error = zero_one_loss(target_test, prediction)
            test_error.append(t_error)
            print(kern, c, " test:", t_error)
        plt.plot(range(len(Cs)), errors, label=kern+ ' train')
        plt.plot(range(len(Cs)), test_error, label=kern+ ' test')
    legend = plt.legend(loc='upper center', shadow=True)
    plt.xticks(np.arange(7), Cs)
    plt.title(title)
    plt.xlabel('C')
    plt.ylabel('error')
    plt.show()
    
test_kernels(input_train, target_train, '2b train')
'''
Which SVM kernel performs best on the test data?
For a large C : sigmoid
'''

######## 2c ###############
test_kernels(input_test, target_test, '2b test')
'''
What bevaviour do you observe now?
Error increas in comprahension to train error TODO
'''

