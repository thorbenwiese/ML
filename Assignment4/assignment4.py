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


def aufg1a():
    gender = vaccination['gender'].tolist()
    gender_count = {x:gender.count(x) for x in gender}
    print('gender groups: ', gender_count)
    age = vaccination['age'].tolist()
    age_count = {x:age.count(x) for x in age}
    print('age groups: ', age_count)
    siblings = vaccination['olderSiblings'].tolist()
    sibling_count = {x:siblings.count(x) for x in siblings}
    print('older siblings: ', sibling_count)
        
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
    return ax1, ax2, ax3, gender_bar, age_bar, sibling_bar


# https://matplotlib.org/examples/api/barchart_demo.html
def autolabel(rects, ax):
    """
    Attach a text label above each gender_bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%d' % int(height),
                ha='center', va='bottom')


'''
b. We are interested in the marginal probabilities of individual values in our data. More
technically, we are interested in P (A = a), where a is a specific value of a random variable A.
The random variables correspond to the fields / column names in the data-set, for example,
A = gender and a = 1 (where 1 denotes “male”). We use short-hand P (a) for P (A = a).
P (a) can be estimated from the data using relative frequencies as follows:
P̂ = rows with a/ all rows
P̂ (a) denotes the empirical estimator of P (a) according to the data.
Calculate the empirical probabilities
– to have a vaccination against disease X,
– to live on the country side,
– to have at least one older sibling.
'''


def marginal_p():
    allPersons = len(vaccination)
    print('P(vaccination against disease X = 1) = ', len(np.where(vaccination['vacX'] == 1)[0]) / allPersons)
    print('P(residence = country side) = ', len(np.where(vaccination['residence'] == 1)[0]) / allPersons)
    print('P(older sibling > 0) = ', len(np.where(vaccination['olderSiblings'] > 0)[0]) / allPersons)


'''
c. Preprocessing variables can help to better understand the data. A common preprocessing
step is to discretize continuous variables. For example, the variable height can be trans-
formed into a binary variable isTallerThan1Meter.
Calculate the following empirical probabilities:
– What is the probability to be taller than 1 meter?
– What is the probability to be heavier than 40 kg?
Another preprocessing step is the combination of variables. Calculate a variable diseaseYZ
which denotes whether a child has had either disease Y or Z or both of them. What is
P̂ (diseaseY Z)?
'''


def preprocessing():
    allPersons = len(vaccination)
    print('P(height > 100) = ', len(np.where(vaccination['height'] > 100)[0]) / allPersons)
    print('P(weight > 40) = ', len(np.where(vaccination['weight'] > 40)[0]) / allPersons)
    print('P(disease X or Y = 1) = ', len(np.where((vaccination['diseaseX'] == 1) | (vaccination['diseaseY'] == 1))[0]) / allPersons)


def main():
    ax1, ax2, ax3, gender_bar, age_bar, sibling_bar = aufg1a()
    autolabel(gender_bar, ax1)
    autolabel(age_bar, ax2)
    autolabel(sibling_bar, ax3)
    marginal_p()
    preprocessing()
    
    plt.show()
    

if __name__ == "__main__":
  main()

