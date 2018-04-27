# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as sme
'''
Assignment 1
Generated datasets based on known distributions are often the best way to test and understand
new algorithms. Numpy offers a wide range of functions to generate and work with random
numbers.

1a. Read the documentation for the numpy.random functions.
Create arrays of n ∈ [100, 1000, 10000, 100 000] random numbers with uniform distribution.
Plot the raw data, then generate and plot histograms with 10 bins. How do the mean,
minimum and maximum values of the bins (occupation counts) behave?
'''
def uniform(k):
    return np.random.uniform(low=0.0, high=1.0, size=k)

def plot(func, input_set, *args):
    fig = plt.figure(figsize=(10, 10))
    fig.canvas.set_window_title(func.__name__)
    outer = gridspec.GridSpec(len(input_set)//2+1, 2, wspace=0.2, hspace=0.2)
    
    for i in range(len(input_set)):
        inner = gridspec.GridSpecFromSubplotSpec(1, 3,
                        subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        s= func(input_set[i], *args)
        title = input_set[i]

        ax = plt.Subplot(fig, inner[0])
        ax.set_title('n = {0:d}'.format(title))
        ax.hist(s, bins=10)
        fig.add_subplot(ax)
        ax1 = plt.Subplot(fig, inner[1])
        #ax1.set_title('k = {0:d}'.format(k))
        ax1.plot(s)
        fig.add_subplot(ax1)
        ax2 = plt.Subplot(fig, inner[2])
        #ax2.set_title('k = {0:d}'.format(k))
        ax2.scatter(s, range(len(s)))
        fig.add_subplot(ax2)
    
# plot(uniform, ([100, 1000, 10000, 100000]))
        
'''
1b. Create random numbers from a Gaussian distribution with mean μ and variance σ2. Plot
the raw data, then generate and plot histograms.
'''  
def gauss(k, args):
    mu, sigma = args[0], args[1]
    return np.random.normal(mu, sigma, k)
# plot(gauss, [100, 1000, 10000, 100000], (0, 0.1))
    
'''
1c. As before, but using the Binomial distribution with parameters n and p.
'''
def binomial(k, args):
    n, p = args[0], args[1]
    return np.random.binomial(n, p, k)

'''
1d. Maybe combining multiple random numbers is even better than using single ones?
Use numpy to generate new random numbers from a sum of individual numbers, si =
PMj=1 rj , where the rj are generated from a uniform distribution. Plot scatter plots and
histograms of the resulting data sets for M ∈ [2, 3, 5, 10, 20].
'''
def individual(k, args):
    a = np.zeros(args)
    for j in range(k):
        b = np.random.uniform(low=0.0, high=1.0, size=args)
        #b.sort() # TODO welche Version richtig?
        a = a + b
    return a

def individual2(k, args):
    a = np.zeros(args)
    for j in range(k):
        b = np.random.uniform(low=0.0, high=1.0, size=args)
        b.sort() # TODO welche Version richtig?
        a = a + b
    return a
#Verison 1: Werte des Arrays der Normalverteilung sind nicht sortiert -> es kommt eine Art Normalverteilung heraus
#Version 2: Arrays sind sortiert -> eine gute Gleichverteilung wird ggf schneller erziehlt? TODO (vergleiche hierfür uniform k=20000 mit individual n=20 (20x k=1000)
 
   
'''
1e. Generate random numbers with a uniform distribution in a circle of radius r.
(Recent versions of numpy actually have a function for this, but the goal here 
is to understand the issue first and then to come up with your own solution.)
'''     
def plotRandomCircle(radius):
  x = []
  y = []
  for i in range(0,1000):
      angle = np.random.uniform(0,1) * 2 * np.pi
      x.append(np.cos(angle)*radius)
      y.append(np.sin(angle)*radius)
  fig = plt.figure()
  plt.scatter(x,y)
  plt.axes().set_aspect('equal', 'datalim')
  plt.title('Circle of random numbers with radius ' + str(radius))
  fig.canvas.set_window_title('circle')
  
def plotRandomCircle2(radius):
  x = []
  y = []
  for i in range(0,1000):
      angle = np.random.uniform(0,1) * 2 * np.pi
      r = np.random.uniform(0,1) # random factor for radius
      x.append(np.cos(angle)*radius*r)
      y.append(np.sin(angle)*radius*r)
  fig = plt.figure()
  plt.scatter(x,y)
  plt.axes().set_aspect('equal', 'datalim')
  plt.title('Circle of random numbers with radius ' + str(radius))
  fig.canvas.set_window_title('circle')
  
#TODO auf Radius oder innerhalb Radius -> auf Kreisaußengrenzen oder innerhalb des Kreis

'''
Assignment 2
Load the data from Adot.mat. Each column of matrix X represents on data point.

2a. Use the function scipy.io.loadmat to parse and load the Matlab/Octave.mat 
data file, then access the array(s) inside the data structures.
'''
def loadMatFile():
  data = sio.loadmat('Adot.mat')
  print( 'X: ', data['X'])
  print( 'Version: ', data['__version__'])
  print( 'Header: ', data['__header__'])
  print( 'Globals: ', data['__globals__'])
  return( data['X'])

'''
2b. Create a numpy matrix for the linear mapping V :
theta = pi/3
V = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
Apply the linear mapping on X to get Y = V X. Plot both X and Y in the same 
figure. What does the linear mapping V do?

2c. Now apply the transpose of the linear mapping on Y to get Z = V^t*Y.
Plot Z and describe what the linear mapping V^t*V does.

2d. What do the linear mappings D1 = [[2, 0], [0, 2]] and D2 = [[2, 0], [0, 1]] 
do? Apply them on X and plot the results.

2e. What does the linear mapping A = V t ∗ D2 ∗ V do? Apply it on X and plot 
the result.
'''
def createLinearMapping(X):
  # b)
  theta = np.pi / 3
  V = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
  Vt = np.transpose(V)
  Y = np.dot(V,X)
  fig = plt.figure()
  plt.title('X and Y with linear mapping V')
  fig.canvas.set_window_title('Linear Mapping of X and Y')
  plt.plot(X,Y)
  # TODO what does V do?

  # c)
  fig = plt.figure()
  plt.title('Z with linear mapping V')
  fig.canvas.set_window_title('Linear Mapping of Z')
  Z = np.dot(Vt,Y)
  plt.plot(Z)
  # what does V^tV do? --> [[ 1.  0.]
  #                         [ 0.  1.]]
  # ==> Identity matrix. Multiplying any matrix with it will result in the 
  #     original matrix

  # d)
  D1 = [[2,0],[0,2]]
  D2 = [[2,0],[0,1]]
  fig = plt.figure()
  fig.canvas.set_window_title('Linear Mapping of X with D1 and D2')
  plt.subplot(2,1,1)
  plt.title('X with linear mapping D1')
  plt.plot(X,np.dot(D1,X))
  plt.subplot(2,1,2)
  plt.title('X with linear mapping D2')
  plt.plot(X,np.dot(D2,X))
  # TODO Interpretation: what does D1 and D2 do?

  # e)
  A = np.dot(np.dot(Vt,D2),V)
  fig = plt.figure()
  fig.canvas.set_window_title('Linear Mapping of X with A')
  plt.title('X with linear mapping A')
  plt.plot(X,A)
  # TODO what does A do?
  
'''
3.
In this exercise, we use a kNN classifier to classify handwritten digits from the USPS data-set. You
can reuse your kNN classifier from Assignment 2 or use libraries from Scikit. The USPS data-set
contains grayscale handwritten digit images scanned from envelopes by the U.S. Postal Service.
The images are of size 16 × 16 (256 pixel) with pixel values in the range 0 to 255. We have 10
classes {1, 2, ..., 9, 0}. The training data has 10000 images, stored in a 10000 × 256 Matlab matrix
(usps train.mat). The training label is a 10000 × 1 vector revealing the labels for the training data.
There are 1000 test images for evaluating your algorithm in the test data (usps test.mat).
'''
    
'''
a. First, we want to classify digit 2 versus digit 3. Prepare the training data: Load the train
data (scipy.io.loadmat) and prepare the training set for classes 2 and 3. We need to convert
the data type from uint8 (8-bit unsigned integer) to double. Do the same for the test data.
'''
def loadData():
    train  = sio.loadmat('usps/usps_train.mat')
    test  = sio.loadmat('usps/usps_test.mat')
    test_data = test['test_data'].astype(float)
    test_label = test['test_label'].astype(float)
    train_data = train['train_data'].astype(float)
    train_label = train['train_label'].astype(float)
    print(test)
    return test_data, test_label, train_data, train_label

def create_classifier(digits, k):
    test_data, test_label, train_data, train_label = loadData()
    indices = [x for x in range(len(test_label)) if test_label[x] in digits]
    d = test_data[indices[0]:indices[-1]]
    dl = test_label[indices[0]:indices[-1]].ravel() # ravel für 2D zu 1D
    print(d)
    print(dl)
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(d, dl)
    return classifier

'''
b. Plot a few example images using matplotlib.pyplot.imshow and the grayscale colormap
(cmap=’grey’ ). Use reshape to convert the image vectors into 16 × 16 images.
'''
def get_image(data, label, index):
    return data[index].reshape(16,16), label[index]

def show_single_image(data, label, index):
    img, l = get_image(data, label, index)
    print(img)
    plt.title('Interpretation: '+str(l))
    plt.imshow(img, cmap='Greys')
    plt.show()
    
def plot_images(images, labels):
    fig = plt.figure(figsize=(10, 10))
    fig.canvas.set_window_title('Images')
    outer = gridspec.GridSpec(len(images)//2+1, 2)
    outer.update(hspace=0.5)
     
    for i in range(len(images)):
        inner = gridspec.GridSpecFromSubplotSpec(1, 1,
                        subplot_spec=outer[i], wspace=0, hspace=0)
 
        ax = plt.Subplot(fig, inner[0])
        ax.set_title('n = {0:d}'.format(int(labels[i][0])))
        ax.imshow(images[i], cmap='Greys')
        fig.add_subplot(ax)
    plt.show()

def show_random_images(data, label, number, singleplots):
    assert len(data) >= number > 0
    selection = np.random.choice(range(len(data)), number, replace=False)
    if(singleplots):
        for x in selection:
            show_single_image(data, label, x)
    else:
        images = []
        labels = []
        for x in selection:
            img, l = get_image(data, label, x)
            images.append(img)
            labels.append(l)
        plot_images(images, labels)

'''
c. Evaluate the performance of your classifier: Test your classifier with different values k =
1, 3, 5, 7, 10, 15 and plot the training and the test errors.
'''
def errorrate(classifier, train_data, train_label, test_data, test_label):
    prediction_train = classifier.predict(train_data)
    prediction_test = classifier.predict(test_data)
    train_error = 1- sme.accuracy_score(train_label, prediction_train)
    test_error = 1- sme.accuracy_score(test_label, prediction_test)
    return train_error, test_error
    
def generate_classifiers(digits, list):
    return [create_classifier(digits, k) for k in list] 

def test_ks(digits, klist, train_data, train_label, test_data, test_label): 
    train = []
    test = []
    for classifier in generate_classifiers(digits, klist):
         train_error, test_error = errorrate(classifier, train_data, train_label, test_data, test_label)
         train.append(train_error)
         test.append(test_error)
         print(train_error, test_error)
    fig = plt.figure()
    plt.xlabel('k')
    plt.ylabel('score in %')
    plt.xticks(range(len(klist)), klist)
    plt.plot(train, label='train score')
    plt.plot(test, label='test score')
    plt.legend()
    plt.show()
        
        
'''
d. Now you can classify other digits. Run your algorithm to classify digit 3 from 8 and compare
its performance with results from digit 2 versus 3.
'''
    # Aufruf von test_ks(digits, klist, train_data, train_label, test_data, test_label) mit anderen Werten

def main():
#     plot(uniform, ([100, 1000, 10000, 100000]))
#     plot(gauss, [100, 1000, 10000, 100000], (0, 0.1)) # (0,0.1) -> (mean, variance)
#     plot(binomial, [100, 1000, 10000, 100000], (10, 0.5)) # ((10, 0.5) -> (n, p)
#     plot(individual, [2, 3, 5, 10, 20], 1000) # 1000 -> größe der uniform distributions die aufaddiert werden)
#     plotRandomCircle(5)
#     plotRandomCircle2(5)
#  
#     X = loadMatFile()
#     createLinearMapping(X)
#  
#     plt.show()
    test_data, test_label, train_data, train_label = loadData()
#     show_random_images(test_data, test_label, 10, False)
#     c23k5 = create_classifier([2,3],5)
#     errorrate(c23k5, train_data, train_label, test_data, test_label)
    test_ks([2,3], [1, 3, 5, 7, 10, 15], train_data, train_label, test_data, test_label)
    test_ks([3,4,5,6,7,8], [1, 3, 5, 7, 10, 15], train_data, train_label, test_data, test_label)
    

if __name__ == "__main__":
  main()
