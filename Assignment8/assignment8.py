# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import scipy.sparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

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

d)
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

print '-' * 25
print 'Assignment 8.2 a) - c)'
print '-' * 25
print ''

def text_classification(count_vect):

  '''
  Aufgabe 8.2
  '''
  categories = ['alt.atheism', 'soc.religion.christian',
	    'comp.graphics', 'sci.med']
  
  twenty_train = fetch_20newsgroups(categories=categories,
                                    shuffle=True,
                                    random_state=41)
  
  texts = twenty_train.data
  labels = twenty_train.target
  names = twenty_train.filenames
  
  print 'Number of loaded texts: ', len(names)
  print ''
  
  X_train_counts = count_vect.fit_transform(twenty_train.data)
  n_samples, n_features = X_train_counts.shape

  print 'Number of words found: ', str(n_features)
  print 'Access to dictionary through count_vect.vocabulary_'
  print 'Index of \'disorganized\': ', count_vect.vocabulary_['disorganized']
  print ''
  
  tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
  X_train_tf = tf_transformer.transform(X_train_counts)
  
  tfidf_transformer = TfidfTransformer()
  X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
  
  clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
  
  docs_new = ['God is love', 'OpenGL on the GPU is fast', 'Machine Learning is fun', 'The winner takes it all']
  X_new_counts = count_vect.transform(docs_new)
  X_new_tfidf = tfidf_transformer.transform(X_new_counts)
  
  predicted = clf.predict(X_new_tfidf)
  
  for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
  print ''
  
  text_clf = Pipeline([('vect', count_vect),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB())])
  
  text_clf.fit(twenty_train.data, twenty_train.target)
  
  twenty_test = fetch_20newsgroups(subset='test',
    categories=categories, shuffle=True, random_state=42)
  docs_test = twenty_test.data
  predicted = text_clf.predict(docs_test)
  print 'Accuracy: ', str(np.mean(predicted == twenty_test.target))
  print ''


cv = CountVectorizer()
text_classification(cv)

print '-' * 20
print 'Assignment 8.2 d)'
print '-' * 20
print '''
The stop_words parameter defines which words are removed from the bag of words 
in order to not tokenize very frequent and meaningless words, such as \'a\', 
\'an\', \'or\', \'it\' and so on.
'''

cve = CountVectorizer(stop_words='english')
text_classification(cve)

print '-' * 20
print 'Assignment 8.2 e)'
print '-' * 20
print '''
Occurences might favor longer documents because they have higher word 
counts. Frequencies show the relative count and are therefore a more balanced 
approach.
'''
