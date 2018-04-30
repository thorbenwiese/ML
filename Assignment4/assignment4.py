import numpy as np
import matplotlib.pyplot as plt

'''
Assignment 04.1 (Probabilities, 1+2+2+2+3 points)
In this exercise, we analyze a simple artificial data-set on vaccination of children. A description
of the data is provided in the file vaccination.readme.txt.

a. Read the vaccination.csv data into your Python workspace. Determine the numbers of
boys/girls, age groups and olderSiblings. Visualize these numbers with gender_bar plots.
'''

vaccination = np.genfromtxt('vaccination.csv', delimiter=',', names=True)
gender = vaccination['gender'].tolist()
gender_count = {x:gender.count(x) for x in gender}
print('gender groups: ', gender_count)
age = vaccination['age'].tolist()
age_count = {x:age.count(x) for x in age}
print('age groups: ', age_count)
siblings = vaccination['olderSiblings'].tolist()
sibling_count = {x:siblings.count(x) for x in siblings}
print('older siblings: ', sibling_count)

print([x for x in gender_count.keys()], [x for x in gender_count.values()])
    
fig1, ax1 = plt.subplots()
ax1.set_title('gender')
ax1.set_ylabel('count')
ax1.set_xticks([x for x in gender_count.keys()])
gender_bar = ax1.bar([x for x in gender_count.keys()], [x for x in gender_count.values()])

fig2, ax2 = plt.subplots()
ax2.set_title('age')
ax2.set_ylabel('count')
ax2.set_xticks([x for x in age_count.keys()])
age_bar = ax2.bar([x for x in age_count.keys()], [x for x in age_count.values()])

fig3, ax3 = plt.subplots()
ax3.set_title('older siblings')
ax3.set_ylabel('count')
ax3.set_xticks([x for x in sibling_count.keys()])
sibling_bar = ax3.bar([x for x in sibling_count.keys()], [x for x in sibling_count.values()])

#https://matplotlib.org/examples/api/barchart_demo.html
def autolabel(rects, ax):
    """
    Attach a text label above each gender_bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(gender_bar, ax1)
autolabel(age_bar, ax2)
autolabel(sibling_bar, ax3)
    
plt.show()

