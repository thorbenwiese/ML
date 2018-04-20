import numpy as np

'''
a. Write your own Python function to generate the Hilbert matrices Hk of dimension k. Using
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
b.  Calculate the rank and condition numbers for the Hilbert matrices with different k. Print
those numbers for k ∈ [1, 30].s
'''
def rank_cond(matrix):
    rank = np.linalg.matrix_rank(matrix)
    condition = np.linalg.cond(matrix)
    print("rank {0:d},  condition: {1:f}".format(rank, condition))

#rank_cond(hilbert(4))

'''
c. Use numpy.linalg.slove() to solve the linear equations Hk · x = b, where b = (1, . . . 1) is a
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
d. Do you trust the solutions?
'''
#TODO

'''
e. In your own words, what is special about these matrices?
'''
#TODO