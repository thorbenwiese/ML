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
