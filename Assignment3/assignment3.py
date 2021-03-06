# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier as classifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
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
        a = a + b
    return a

def individual2(k, args):
    a = np.zeros(args)
    for j in range(k):
        b = np.random.uniform(low=0.0, high=1.0, size=args)
        b.sort()
        a = a + b
    return a
#welche Version richtig?
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
# # auf Radius oder innerhalb Radius -> auf Kreisaußengrenzen oder innerhalb des Kreis

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
  # What does V do? -> it rotates the vectors to the left and right

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
  # Interpretation: what does D1 and D2 do?
  # --> D1 doubles the values in the matrix
  # --> D2 doubles the first half and keeps the second half

  # e)
  A = np.dot(np.dot(Vt,D2),V)
  fig = plt.figure()
  fig.canvas.set_window_title('Linear Mapping of X with A')
  plt.title('X with linear mapping A')
  plt.plot(X,np.dot(A,X))
  # What does A do? --> it rotates the first and second half in different
  # directions
  
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
    return test_data, test_label, train_data, train_label

def trainModel(num1, num2, train_data, train_label, test_data, test_label, k):
  #ohne Skalierung sind die scores höher.....
  #scaler = StandardScaler()  
  #scaler.fit(train_data)
  #train_data = scaler.transform(train_data)
  #scaler.fit(test_data)
  #test_data = scaler.transform(test_data)

  # Train Data
  train_values = []
  train_labels = []
  for i,x in enumerate(train_label):
    if x == [num1] or x == [num2]:
      train_values.append(train_data[i])
      train_labels.append(x)

  train_labels = np.ravel(train_labels)

  # Test Data
  test_values = []
  test_labels = []
  for i,x in enumerate(test_label):
    if x == [num1] or x == [num2]:
      test_values.append(test_data[i])
      test_labels.append(x)

  test_labels = np.ravel(test_labels)

  knn = classifier(n_neighbors=k)
  knn.fit(train_values, train_labels) 
  predict = knn.predict(test_data)

  print ('Prediction:\n', predict, '\n')
  print ('Prediction probabilities:\n', knn.predict_proba(test_data))

  train_score = knn.score(train_values, train_labels)
  test_score = knn.score(test_values, test_labels)

  print ('Train score: ', train_score)
  print ('Test score: ', test_score)

  return train_score, test_score

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

# def plot_images(images, labels):
#     fig, axes = plt.subplots(nrows=4, ncols=4)
#     fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
#     fig.canvas.set_window_title('Images')
#     
#     for i in range(len(images)):
#         ax = plt.subplot(fig)
#         ax.set_title('n = {0:f}'.format(labels[i][0]))
#         ax.imshow(images[i], cmap='Greys')
#         fig.add_subplot(ax)
#     plt.show()
    
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
    assert len(data) > number > 0
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
d. Now you can classify other digits. Run your algorithm to classify digit 3 from 8 and compare
its performance with results from digit 2 versus 3.
'''



def main():
     plot(uniform, ([100, 1000, 10000, 100000]))
     plot(gauss, [100, 1000, 10000, 100000], (0, 0.1)) # (0,0.1) -> (mean, variance)
     plot(binomial, [100, 1000, 10000, 100000], (10, 0.5)) # ((10, 0.5) -> (n, p)
     plot(individual, [2, 3, 5, 10, 20], 1000) # 1000 -> größe der uniform distributions die aufaddiert werden)
     plotRandomCircle(5)
     plotRandomCircle2(5)
  
     X = loadMatFile()
     createLinearMapping(X)
  
     test_data, test_label, train_data, train_label = loadData()
     show_random_images(test_data, test_label, 10, False)
     trainModel(2, 3, train_data, train_label, test_data, test_label, 5)

     train_scores = []
     test_scores = []
     train_scores2 = []
     test_scores2 = []
     for k in [1, 3, 5, 7, 10, 15]:
       train_score, test_score = trainModel(2, 3, train_data, train_label, test_data, test_label, k)
       train_scores.append(train_score)
       test_scores.append(test_score)

       train_score2, test_score2 = trainModel(3, 8, train_data, train_label, test_data, test_label, k)
       train_scores2.append(train_score2)
       test_scores2.append(test_score2)

     fig = plt.figure()
     plt.title('Train vs. Test Score (2 and 3) for different k')
     plt.xlabel('k')
     plt.ylabel('score in %')
     x_labels = [1,3,5,7,10,15]
     plt.xticks([0,1,2,3,4,5], x_labels)
     plt.plot(train_scores, label='train score')
     plt.plot(test_scores, label='test score')
     plt.legend()

     fig = plt.figure()
     plt.title('Train vs. Test Score (3 and 8) for different k')
     plt.xlabel('k')
     plt.ylabel('score in %')
     x_labels = [1,3,5,7,10,15]
     plt.xticks([0,1,2,3,4,5], x_labels)
     plt.plot(train_scores2, label='train score')
     plt.plot(test_scores2, label='test score')
     plt.legend()

     plt.show()
    

if __name__ == "__main__":
  main()
