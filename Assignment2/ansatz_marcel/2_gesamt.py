import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import math

'''
2.2a. Write your own Python function to generate the Hilbert matrices Hk of dimension k. Using
Python-style (zero-based) addressing, the matrix element in row i and column j has the
value 1/(i + j + 1). Try to use numpy array operations for best performance.
'''
def hilbert(k):
    matrix = np.ones(shape=(k,k), dtype=np.float)
    for index, x in np.ndenumerate(matrix):
        #print(index,x)
        i,j = index[0]+1, index[1]+1
        x = 1 / (i+j-1)
        matrix[index[0]][index[1]] = x
        #print(index,x)
    return np.asmatrix(matrix)

#print(hilbert(4))

'''
2.2b.  Calculate the rank and condition numbers for the Hilbert matrices with different k. Print
those numbers for k ∈ [1, 30].s
'''
def rank_cond(matrix):
    rank = np.linalg.matrix_rank(matrix)
    condition = np.linalg.cond(matrix)
    print("rank {0:d},  condition: {1:f}".format(rank, condition))

#rank_cond(hilbert(4))

'''
2.2c. Use numpy.linalg.slove() to solve the linear equations Hk · x = b, where b = (1, . . . 1) is a
vector of all ones. Print the solutions xk for k ∈ [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]. Check your
solutions by again calculating ||Hk · xk − b|| using numpy
'''
def solve(k, printlist): 
    H = hilbert(k)
    b = np.ones((k,1)) 
    x = np.linalg.solve(H, b)
    check = np.linalg.norm(np.dot(H,x)-b)
    if printlist:
        print("k = {0:d}, x = {1:s}, precision = {2:.2E}".format(k,str(np.transpose(x)), check)) # x transponiert für bessere Lesbarkeit
        #print("k = {0:d}, x = {1:s}, precision = {2:.25f}".format(k,str(np.transpose(x)), check)) 
    else:
        print("k = {0:d}, precision = {1:.2E}".format(k, check))
        
def test_solve(list, printlist):
    for x in list:
        solve(x, printlist)
        
#test_solve([1,2,3,5,10,15,20,30,50,100], printlist=False)

'''
2.2d. Do you trust the solutions?
'''
#TODO

'''
2.2e. In your own words, what is special about these matrices?
'''
#TODO

'''
2.3a. Load the dataset into Python (e.g. using csv reader, numpy, or pandas).
''' 
data = np.genfromtxt('housing.csv', delimiter=',', dtype=None,  names=True)
#print(data)

'''
2.3b. For all data columns, find and print the minimum and maximum values, find and print the
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
2.3c. For all data columns, calculate and plot a histogram of the data. Is the data from a normal
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
2.3d. Generate a geographical map of the houses using a scatter plot based on the longitude and
latitude data. Play with transparency (e.g., use alpha 0.1) for less clutter. Next, use a color
scale (e.g. blue to red) to color-encode the house value in the scatter plot.
'''
plt.figure()
plt.title("Geographical map of houses")
plt.scatter(data['longitude'], data['latitude'], alpha=0.1, c=data['median_house_value'], cmap=cm.get_cmap('RdBu_r'))#red to blue reversed
#alpha = transparent, c = Daten anhand der Farben gesetzt werden, cmap = Farbpalette

'''
2.3e. Split the dataset randomly into a training set (80% of the data points) and a test set (20%
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

'''
2.4a. Write a Python function that calculates the L2 loss for the housing value.
'''
    
np.random.seed(9876)
mod = GradientBoostingRegressor(loss='ls')
print(y_train)
fit = mod.fit(X_train, y_train)
predict = fit.predict(X_test)
loss = mean_squared_error(predict, y_test)   
'''
2.4b Design and implement a meaningful distance function for the housing dataset.
'''
def distance(s1,s2, start, end):
    assert len(s1) == len(s2)
    d = 0
    if(start == -1 & end == -1):
        for i in range(len(s1)):
            d += math.sqrt(pow((s1[i] - s2[i]), 2))
    else:
        for i in range(start, end+1):
            d = math.sqrt(pow((s1[i] - s2[i]), 2))
    return d

def distance_set(set1, set2, mode):
    dist = []
    assert len(set1) == len(set2)
    for i in range(len(set1)):
        dist.append(distance(set1[i], set2[i])) #euklidischer Abstand
    dist.sort()
    if(mode is 'mean'):
        return sum(dist)/ len(dist)
    elif(mode is 'median'):
        return dist[(len(dist))//2]
    elif(mode is 'sum'):
        return sum(dist)
    elif(mode is 'min'):
        return dist[0]
    elif(mode is 'max'):
        return dist[-1]
    else:
        raise Exception('illegal mode')
    
'''
2.4c. Implement the kNN algorithm, using the loss and distance functions above, and using the
training- and test-sets from the previous exercise.
'''
def neighbors(set, instance, k):
    dist = []
    for x in range(len(set)):
        dist.append([x,distance(instance, x, 0,1)]) # longitude + latitude
    dist.sort(key=lambda x: x[1])
    return dist[0:k]
#TODO den Rest drum herum, sprich für potentielle Nachbarn Ort der Instanz als Vorhersage setzen

'''
d. Predict the housing values for different values of k. What are your training and test errors?
'''
def error_rate(instance, prediction):
# Idee: eine Instanz wird mit einer Menge vorhergesagter Nachbarn verglichen, wenn Ort nicht übereinstimmt dann wird Error hochgezählt
    error_count = 0
    for i in range(len(prediction)):
        if(not set[i][-1] is prediction[i][-1]):
            error_count += 1
    return error_count / len(set)

def mean_error_rate():
    pass #TODO





plt.show()
