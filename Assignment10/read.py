from ast import literal_eval as make_tuple
import matplotlib.pyplot as plt

lines = [line.rstrip('\n') for line in open('result.txt')]

readIds = []
readMax = []
readMin = []
readVar = []
readNumPix = []
readShape = []

for line in lines:
    #print line.split(',')
    split = line.split(',')
    if len(split) == 8:
      readIds.append(split[0].replace('(','').replace('.jpg',''))
      readShape.append(split[1] + split[2] + split[3])
      readMax.append(split[4])
      readMin.append(split[5])
      readVar.append(split[6])
      readNumPix.append(split[7].replace(')',''))
    elif len(split) == 7:
      readIds.append(split[0].replace('(','').replace('.jpg',''))
      readShape.append(split[1] + split[2])
      readMax.append(split[3])
      readMin.append(split[4])
      readVar.append(split[5])
      readNumPix.append(split[6].replace(')',''))
    else:
      print split

'''
print readIds
print ''
print readShape
print ''
print readMax
print ''
print readMin
print ''
print readVar
print ''
print readNumPix
'''

plt.figure()
plt.scatter(readMax, range(len(readMax)))
plt.title('readMax')

plt.figure()
plt.scatter(readMin, range(len(readMin)))
plt.title('readMin')

plt.figure()
plt.scatter(readVar, range(len(readVar)))
plt.title('readVar')

plt.figure()
plt.scatter(readNumPix, range(len(readNumPix)))
plt.title('readNumPix')

plt.figure()
plt.scatter(readShape, range(len(readShape)))
plt.title('readShape')


plt.show()
