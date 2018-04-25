# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio
'''
1a. Read the documentation for the numpy.random functions.
Create arrays of n ∈ [100, 1000, 10000, 100 000] random numbers with uniform distribution.
Plot the raw data, then generate and plot histograms with 10 bins. How do the mean,
minimum and maximum values of the bins (occupation counts) behave?
'''
def uniform(i,input_set):
    return np.random.uniform(low=0.0, high=1.0, size=input_set[i]), input_set[i]

def plot(func, input_set, *args):
    fig = plt.figure(figsize=(20, 20))
    fig.canvas.set_window_title(func.__name__)
    outer = gridspec.GridSpec(len(input_set)//2+1, 2, wspace=0.2, hspace=0.2)
    
    for i in range(len(input_set)):
        inner = gridspec.GridSpecFromSubplotSpec(1, 3,
                        subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        s, k= func(i,input_set, *args)

        ax = plt.Subplot(fig, inner[0])
        ax.set_title('n = {0:d}'.format(k))
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
def gauss(i,input_set, args):
    mu, sigma = args[0], args[1]
    return np.random.normal(mu, sigma, input_set[i]), input_set[i]
# plot(gauss, [100, 1000, 10000, 100000], (0, 0.1))
    
'''
1c. As before, but using the Binomial distribution with parameters n and p.
'''
def binomial(i, input_set, args):
    n, p = args[0], args[1]
    return np.random.binomial(n, p, input_set[i]), input_set[i]

'''
1d. Maybe combining multiple random numbers is even better than using single ones?
Use numpy to generate new random numbers from a sum of individual numbers, si =
PMj=1 rj , where the rj are generated from a uniform distribution. Plot scatter plots and
histograms of the resulting data sets for M ∈ [2, 3, 5, 10, 20].
'''
def individual(i, input_set, args):
    a = np.zeros(args)
    for j in range(input_set[i]):
        b = np.random.uniform(low=0.0, high=1.0, size=args)
        #b.sort() # TODO richtig?
        a = a + b
    return a, input_set[i]
   
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

'''
2a. Use the function scipy.io.loadmat to parse and load the Matlab/Octave.mat 
data file, then access the array(s) inside the data structures.
'''
def loadMatFile():
  data = sio.loadmat('Adot.mat')
  print 'X: ', data['X']
  print 'Version: ', data['__version__']
  print 'Header: ', data['__header__']
  print 'Globals: ', data['__globals__']
  return data['X']

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

def main():
    plot(uniform, ([100, 1000, 10000, 100000]))
    plot(gauss, [100, 1000, 10000, 100000], (0, 0.1)) # (0,0.1) -> (mean, variance)
    plot(binomial, [100, 1000, 10000, 100000], (10, 0.5)) # ((10, 0.5) -> (n, p)
    plot(individual, [2, 3, 5, 10, 20], 1000) # 1000 -> größe der uniform distributions die aufaddiert werden)
    plotRandomCircle(5)

    X = loadMatFile()
    createLinearMapping(X)

    plt.show()

if __name__ == "__main__":
  main()
