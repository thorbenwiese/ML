# -*- coding: utf-8 -*-

'''
Aufgabe 8.1
a)

The 7 different matrices are used for different kinds of original sparse 
matrices.

DIA - Diagonal Format
LIL - List of Lists Format
DOK - Dictionary of Keys Format
COO - Coordinate Format
CSR - Compressed Sparse Row Format
CSC - Compressed Sparse Column Format
BSR - Block Compressed Row Format

csr_matrix is the Compressed Sparse Row Format (CSR)
The data is stores as arrays of rows and their values at each column
Very efficient at accessing row-vectors or performing row-operations

b)

The COO Format offers a way to change the rows, columns and values from a 
matrix and is therefore a good choice for inserting nonzero values.

The CSR Format is best suited for multiplying the matrix with a vector or 
another matrix since it is row based and therefore a good choice for matrix 
multiplication.

c)

The scipy.sparse.find(A) functions returns the row indices, column indices and 
values of the nonzero matrix A entries. One can use numpy.ma.size(A) to get all 
elements and then calculate

i, j, v = scipy.sparse.find(A)
sparsity = (len(v) * 1.0) / numpy.ma.size(A)

to get the sparsity percentage.
'''

import numpy as np
import scipy as sp
import scipy.sparse

'''
Aufgabe 8d)
'''
def tridiagonal(n):

  diags = np.array([np.random.randint(10, size=n),
                    np.random.randint(10, size=n),
                    np.random.randint(10, size=n)])
  positions = [-1, 0, 1]
  print sp.sparse.spdiags(diags, positions, n, n).todense()
  print ''

tridiagonal(5)
tridiagonal(10)


'''
Aufgabe 8.2
'''
from sklearn.datasets import fetch_20newsgroups

twenty_train = fetch_20newsgroups(categories=['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian'], shuffle=True, random_state=41)

texts = twenty_train.data
labels = twenty_train.target
names = twenty_train.filenames

print '-' * 20
print 'Assignment 8.2 a)'
print '-' * 20
print ''
print 'Number of loaded texts: ', len(names)
print ''

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
n_samples, n_features = X_train_counts.shape

print '-' * 20
print 'Assignment 8.2 b)'
print '-' * 20
print ''
print 'Number of words found: ', str(n_features)
print 'Access to dictionary through count_vect.vocabulary_'
print 'Index of \'disorganized\': ', count_vect.vocabulary_['disorganized']
