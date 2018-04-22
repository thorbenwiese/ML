import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
'''
a. Load the dataset into Python (e.g. using csv reader, numpy, or pandas).
''' 
data = np.genfromtxt('housing.csv', delimiter=',', dtype=None,  names=True)
#print(data)

'''
b. For all data columns, find and print the minimum and maximum values, find and print the
indices of those values, calculate the mean value.
'''
def analyse(set):
    distributions = []
    for column in set.dtype.names:
        try:
            min_index, max_index = np.argmin(set[column]), np.argmax(set[column])
            min_value, max_value = set[column][min_index], set[column][max_index]
            mean = np.mean(set[column])
            statistic, pvalue  = stats.normaltest(set[column]) # test normaldistribution
            #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
            alpha = 1e-3
            if pvalue < alpha:  # null hypothesis: x comes from a normal distribution
                normal = 'no'
            else:
                normal = 'yes'
            distributions.append([column, normal])
            print("column: {0:s}\n min_index= {1:d}\n min_value= {2:f}\n max_index= {3:d}\n max_value= {4:f}\n mean= {5:f}\n normal distribution= {6:s} \n".format(column, min_index, min_value, max_index, max_value, mean, normal))
        except TypeError:
            print("elements of column {0:s} contains strings".format(column))
    return distributions
analyse(data)

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

'''
d. Generate a geographical map of the houses using a scatter plot based on the longitude and
latitude data. Play with transparency (e.g., use alpha 0.1) for less clutter. Next, use a color
scale (e.g. blue to red) to color-encode the house value in the scatter plot.
'''
plt.figure()
plt.title("Geographical map of houses")
plt.scatter(data['longitude'], data['latitude'], alpha=0.1, c=data['median_house_value'], cmap=cm.get_cmap('RdBu_r'))#red to blue reversed
#alpha = transparent, c = Daten anhand der Farben gesetzt werden, cmap = Farbpalette

'''
e. Split the dataset randomly into a training set (80% of the data points) and a test set (20%
of the data points). Use a fixed random seed, so that your split can be reproduced later. For
both sets, repeat the calculation of the minimum, maximum, and mean values. Does your
test-set match the distribution in the training-set?
'''
X_train, X_test, y_train, y_test = train_test_split(data[1:],range(len(data[1:])), test_size=0.20, random_state= 42)
print('Analyse von X_train\n')
distribution_train = analyse(X_train)
print('-'*100)
print('Analyse von X_test\n')
distribution_test = analyse(X_test)
print('-'*100)
for i in range(len(distribution_test)):
    column = distribution_test[i][0]
    assert distribution_test[i][0] == distribution_train[i][0]
    match = (distribution_test[i][1] == distribution_train[i][1])
    print('{0:s} : distribution match -> {1:}'.format(column, match))





plt.show()