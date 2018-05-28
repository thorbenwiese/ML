import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import random
import timeit
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import zero_one_loss
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

######## 1b ###############
'''
How do you configure and train the SVM
There are some attributes for the configuration: penalty parameter c of the error term, kernel, ...
We used the default configuration
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
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
fignum = 1
data = sio.loadmat('cancer-data.mat')
input_train = data['cancerInput_train']
target_train = data['cancerTarget_train'].flatten()
input_test = data['cancerInput_test']
target_test = data['cancerTarget_test'].flatten()

Cs = [0.01, 0.1, 0.5, 1, 5, 10, 50]
errors =[]
test_error = []
for c in Cs:
    clf = LinearSVC(C=c, class_weight=None, dual=True, fit_intercept=True, #TODO linear oder anderen?
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
         verbose=0)
    model = clf.fit(input_train, target_train)
    #error = 1- model.score(input_train, target_train)
    error = zero_one_loss(target_train, clf.predict(input_train))
    errors.append(error)
    prediction = clf.predict(input_test)
    t_error = zero_one_loss(target_test, prediction)
    test_error.append(t_error)
    print(c, " train:", error) #TODO richtig für train error?
    print(c, " test:", t_error)
plt.figure(fignum)
plt.plot(range(len(Cs)), errors, label ='train_error')
plt.plot(range(len(Cs)), test_error, label='test_error')
plt.legend(loc='upper center', shadow=True)
plt.ylim(0, 0.05)
plt.xticks(np.arange(7), Cs)
plt.xlabel('C')
plt.ylabel('error')
plt.title('2a')
fignum += 1
    
'''
What is the effect of choosing a large C on the training error?
Choosing a large C results in a decreasing training error.

Does this effect coincide with what you are expecting?
Yes, because C is the penalty parameter of the error term. So a increasing C tells the SVM optimization 
how much you want to avoid misclassifying each training example. TODO outlier werden betrachtet
'''


######## 2b ###############
#TODO (gridsearchcv,  crossvalidation -> ) auch Werte für gamma und degree -> zb gamma 3-4 plots
#https://chrisalbon.com/machine_learning/model_evaluation/cross_validation_parameter_tuning_grid_search/
def test_kernels(t, t_label, plottrain, title):
    plt.figure(fignum)
    for kern in ['linear', 'poly', 'rbf', 'sigmoid']:#, 'precomputed']:
        errors = []
        test_error = []
        for c in Cs:
            clf = SVC(C=c, kernel=kern, degree=3, gamma='auto', coef0=0.0, shrinking=True, 
                      probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, 
                      max_iter=-1, decision_function_shape='ovr', random_state=None)
            model = clf.fit(t, t_label)
            #error = 1- model.score(t, t_label)
            error = zero_one_loss(target_train, clf.predict(input_train))
            errors.append(error)
            print(kern, c, "train:", error)
            prediction = clf.predict(input_test)
            t_error = zero_one_loss(target_test, prediction)
            test_error.append(t_error)
            print(kern, c, " test:", t_error)
        if plottrain:
            plt.plot(range(len(Cs)), errors, label=kern+ ' train')
        plt.plot(range(len(Cs)), test_error, label=kern+ ' test')
    plt.legend(loc='upper center', shadow=True)
    plt.xticks(np.arange(7), Cs)
    plt.title(title)
    plt.xlabel('C')
    plt.ylabel('error')
    global fignum
    fignum +=1
    
test_kernels(input_train, target_train, True, '2b')
'''
Which SVM kernel performs best on the test data?
TODO -> linear schlechter als andere -> rbf best
'''

######## 2c ###############
test_kernels(input_test, target_test, True, '2c')
'''
What bevaviour do you observe now?
Test Error increase in comparison to train error
'''
plt.show()

######## 3 ###############
usps_test = sio.loadmat('usps_test.mat')
test_data = usps_test['test_data']
test_label = usps_test['test_label'].flatten()
usps_train = sio.loadmat('usps_train.mat')
train_data = usps_train['train_data']
train_label = usps_train['train_label'].flatten()

reg= LinearRegression()
reg.fit(train_data,train_label)
#print(reg.predict(test_data))
#print(test_label)

def knn(d, n, k):
    ran = range(0,len(train_data))
    sample = random.sample(ran, n)
    t = [train_data[x][:d] for x in sample]
    tl = [train_label[x] for x in sample]
    start = timeit.default_timer()
    knn = KNeighborsRegressor(k)
    knn.fit(t, tl)
    b = [test_data[x][:d] for x in range(0,len(test_data))]
    knn.predict(b)
    #knn.predict(test_data)
    stop = timeit.default_timer()
    print ('knn d=',d,' n=',n,' k=',k, ' ->' , stop - start)
    
knn(1, 1000, 5)
for n in [100,500, 1000, 3000, 5000, 7000, 10000]:
    for k in [1,2,3,5,10]:
        for d in [1,10,100,255]:
            knn(d, n, k)
        
#clf = SVC(decision_function_shape='ovr') # ovr = one-vs-rest,  ovo = one-vs-one
'''
The estimated time values should be in O(nd+kn)


What is the computational complexity of predicting a new data point for a SVM?  (m support
vectors after training)

Linear SVM has prediction complexity O(d) with d the number of input dimensions since it is 
just a single inner product. Prediction complexity of kernel SVM depends on the choice of 
kernel and is typically proportional to the number of support vectors. For most kernels,
including polynomial and RBF, this is O(nSVd) where nSV is the number of support vectors.


How much information do you need to store for predicting with each of these methods (space
complexity)?
Knn needs O(nd), SVM needs O(md)

For a specific example, consider a one-against-the-rest classifier for the full USPS dataset
(d = 256, n = 10000), assume k = 10 for kNN and m = 1000 support vectors for the SVM
classifier. How many operations and how much memory is needed?

operations knn: O(nd + kn) = 10000 * 256 + 10 * 10000 = 2660000
memory knn: O(nd) = 10000 * 256 = 2560000
operations SVM: O(nSVd) = O(md) = 1000 * 256 = 256000
memory SVM: O(md) = 1000 * 256 = 256000


'''

