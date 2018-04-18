import numpy as np

print np.__version__

print np.arange(10)

print np.full((3,3), True, dtype=bool)

arr = np.arange(10)

print arr[arr % 2 == 1]

arr[arr % 2 == 1] = -1
print arr

arr = np.arange(10)
print [x if x % 2 == 0 else -1 for x in arr]
print np.where(arr % 2 == 1, -1, arr)
print arr


print np.arange(10).reshape((2,5))

print np.vstack((np.arange(10).reshape(2,-1), np.repeat(1,10).reshape(2,-1)))

print np.concatenate((np.arange(10).reshape(2,-1), np.repeat(1,10).reshape(2,-1)), 1)


a = np.array([1,2,3])

print np.concatenate((np.repeat(a, 3), np.tile(a,3)),1)

a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])

print np.intersect1d(a,b)

a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
print np.setdiff1d(a, b)

a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])

print (np.where(a == b))

a = np.array([2, 6, 1, 9, 10, 3, 27])
print a[np.where(np.logical_and(a > 4, a < 11))]
