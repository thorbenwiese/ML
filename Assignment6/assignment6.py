import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

######## 1a ###############
data = sio.loadmat('iris_multiclass.mat')
species = data['species']
meas = data['meas']
indices_train = data['indices_train'][0]-1 # -1 da Indizes bei 1 starten -> Normalisierung
indices_test = data['indices_test'][0]-1 # -1 da Indizes bei 1 starten -> Normalisierung
train = [(x,meas[x], species[x][0][0]) for x in indices_train]
test = [(x,meas[x], species[x][0][0]) for x in indices_test]
print(test[0])




######### 1c ####################
'''
scipy.optimize.leastsq¶

Scipy provides a method called leastsq as part of its optimize package. However, there are tow problems:

    - This method is not well documented (no easy examples).
    - Error/covariance estimates on fit parameters not straight-forward to obtain.

Internally, leastsq uses Levenburg-Marquardt gradient method (greedy algorithm) to minimise the score function.
'''