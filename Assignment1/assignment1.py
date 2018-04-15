import numpy as np
import matplotlib.pyplot as plt

# read data from file
input_data = np.genfromtxt("traffic_per_hour.csv")
# remove NaN values from data
data = input_data[np.logical_not(np.isnan(input_data[:,1]))]

# split data into indices and values
indices = data[:,0]
values = data[:,1]

# define range for plotting
r = range(1,10000)

# plot settings
plt.title('website accesses per hour')
plt.xlabel('time in hours')
plt.xlim(-50,1500)
plt.ylabel('website accesses')
plt.ylim(-50,12500)

# create scatter plot of data points
plt.scatter(indices, values, color='black', s=2)

# plot first 10 dimensional polynoms fitting the scattered data
for i in range(1,10):
  func = np.poly1d(np.polyfit(indices, values, i))
  plt.plot(r, func(r), label=str(i) + ' dim')

# plot x values for y=10000
plt.plot([600,1300],[10000,10000], color='black', linestyle=':', 
label='y = 10000')

# plot legend
plt.legend()

# show plot
plt.show()
