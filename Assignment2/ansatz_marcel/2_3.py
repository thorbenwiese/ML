import numpy as np
import matplotlib.pyplot as plt 
'''
a. Load the dataset into Python (e.g. using csv reader, numpy, or pandas).
''' 
data = np.genfromtxt('housing.csv', delimiter=',', dtype=None,  names=True)
#print(data)

'''
b. For all data columns, find and print the minimum and maximum values, find and print the
indices of those values, calculate the mean value.
'''
for column in data.dtype.names:
    try:
        min_index, max_index = np.argmin(data[column]), np.argmax(data[column])
        min_value, max_value = data[column][min_index], data[column][max_index]
        mean = np.mean(data[column])
        print("column: {0:s}\n min_index= {1:d}\n min_value= {2:f}\n max_index= {3:d}\n max_value= {4:f}\n mean= {5:f}\n\n".format(column, min_index, min_value, max_index, max_value, mean))
    except TypeError:
        print("elements of column {0:s} contains strings".format(column))


'''
c. For all data columns, calculate and plot a histogram of the data. Is the data from a normal
distribution?
'''
f,a = plt.subplots(3,3)
a = a.ravel()
for idx,ax in enumerate(a):
    print(data[data.dtype.names[idx]])
    col = data[data.dtype.names[idx]]
    col = col[~np.isnan(col)] # remove nan
    ax.hist(col)
    ax.set_title(data.dtype.names[idx])
plt.tight_layout()
plt.show()
#TODO normal distribution? TODO