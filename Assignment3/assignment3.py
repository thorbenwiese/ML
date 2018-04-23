import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
'''
a. Read the documentation for the numpy.random functions.
Create arrays of n ∈ [100, 1000, 10000, 100 000] random numbers with uniform distribution.
Plot the raw data, then generate and plot histograms with 10 bins. How do the mean,
minimum and maximum values of the bins (occupation counts) behave?
'''

def plot_uniform(set):
    for i in set:
        #mu, sigma = 0, 0.1 
        #s = np.random.normal(mu, sigma, i)
        s= np.random.uniform(low=0.0, high=1.0, size=i)
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.hist(s, bins=10)
        ax2= fig.add_subplot(122)
        ax2.plot(s)
        fig.suptitle(i)
    plt.show()
    
def plot_uniform2(set):
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(len(set)//2+1, 2, wspace=0.2, hspace=0.2)
    
    for i in range(len(set)):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                        subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        
        #mu, sigma = 0, 0.1 
        #s = np.random.normal(mu, sigma, i)
        s= np.random.uniform(low=0.0, high=1.0, size=set[i])

        ax = plt.Subplot(fig, inner[0])
        ax.set_title('k = {0:d}'.format(set[i]))
        ax.hist(s, bins=10)
        fig.add_subplot(ax)
        ax1 = plt.Subplot(fig, inner[1])
        ax1.set_title('k = {0:d}'.format(set[i]))
        ax1.plot(s)
        fig.add_subplot(ax1)

    plt.show()

def uniform(x,i):
    return np.random.uniform(low=0.0, high=1.0, size=x[i]), x[i]

def plot(func, *set):
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(len(*set)//2+1, 2, wspace=0.2, hspace=0.2)
    
    for i in range(len(*set)):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                        subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        s, k= func(*set, i)

        ax = plt.Subplot(fig, inner[0])
        ax.set_title('k = {0:d}'.format(k))
        ax.hist(s, bins=10)
        fig.add_subplot(ax)
        ax1 = plt.Subplot(fig, inner[1])
        #ax1.set_title('k = {0:d}'.format(k))
        ax1.plot(s)
        fig.add_subplot(ax1)

    plt.show()
    

    
'''
b. Create random numbers from a Gaussian distribution with mean μ and variance σ2. Plot
the raw data, then generate and plot histograms.
'''
    
def plot_gauss(set, mean, sigma): 
    #sigma== standard deviation
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(len(set)//2+1, 2, wspace=0.2, hspace=0.2)
    
    for i in range(len(set)):
        inner = gridspec.GridSpecFromSubplotSpec(1, 3,
                        subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        
        s = np.random.normal(mean, sigma, set[i])

        ax = plt.Subplot(fig, inner[0])
        ax.set_title('k = {0:d}'.format(set[i]))
        ax.hist(s, bins=10)
        fig.add_subplot(ax)
        ax1 = plt.Subplot(fig, inner[1])
        ax1.set_title('k = {0:d}'.format(set[i]))
        ax1.plot(s)
        fig.add_subplot(ax1)
        ax2 = plt.Subplot(fig, inner[2])
        ax2.set_title('k = {0:d}'.format(set[i]))
        ax2.scatter(s, range(len(s)))
        fig.add_subplot(ax2)
    plt.show()
    
'''
c. As before, but using the Binomial distribution with parameters n and p.
'''



def main():
   #plot_uniform([100, 1000, 10000, 100000]) 
   #plot_uniform2([100, 1000, 10000, 100000])
   #plot_gauss([100, 1000, 10000, 100000], 0, 0.1)
   plot(uniform, [100, 1000, 10000, 100000])

if __name__ == "__main__":
  main()