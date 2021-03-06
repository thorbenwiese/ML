# -*- coding: utf-8 -*-
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
    
    autolabel(gender_bar, ax1)
    autolabel(age_bar, ax2)
    autolabel(sibling_bar, ax3)


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

'''
 Conditional probabilities relate two or more variables. P (a | b) measures the probabil-
ity of a given that we know b. For example, P (diseaseX = 1 | vacX = 0) quantifies the
probability that someone has had disease X given that he/she was not vaccinated against
X.
P (a | b) can be estimated using relative frequencies as follows:
P̂ (a | b) = rows with a and b / rows with b
Calculate the following probabilities:
– P̂ (diseaseX | vacX = 0/1)
– P̂ (vacX | diseaseX = 0/1)
– P̂ (diseaseY | age = 1/2/3/4)
– P̂ (vacX | age = 1/2/3/4)
– P̂ (knowsT oRideABike | vacX = 0/1)
where P̂ (a | b = 0/1) is shorthand for P̂ (a = 1 | b = 0) and P̂ (a = 1 | b = 1).
Visualize P̂ (diseaseY | age = 1/2/3/4) and P̂ (vacX | age = 1/2/3/4) as line plots with age
on the x-axis. What can you conclude from your results?
'''

def conditional_p():   
    print('P(diseaseX = 1 | vacX = 0)\t -> ', 
          len(np.where((vaccination['diseaseX'] == 1) & (vaccination['vacX'] == 0))[0]) / len(np.where(vaccination['vacX'] == 0)[0]))
    print('P(diseaseX = 1 | vacX = 1)\t -> ',
          len(np.where((vaccination['diseaseX'] == 1) & (vaccination['vacX'] == 1))[0]) / len(np.where(vaccination['vacX'] == 1)[0]))
    print('P(vacX = 1 | diseaseX = 0)\t -> ',
          len(np.where((vaccination['vacX'] == 1) & (vaccination['diseaseX'] == 0))[0]) / len(np.where(vaccination['diseaseX'] == 0)[0]))
    print('P(vacX = 1 | diseaseX = 1)\t -> ',
          len(np.where((vaccination['vacX'] == 1) & (vaccination['diseaseX'] == 1))[0]) / len(np.where(vaccination['diseaseX'] == 1)[0]))
    dis_a1 = len(np.where((vaccination['diseaseY'] == 1) & (vaccination['age'] == 1))[0]) / len(np.where(vaccination['age'] == 1)[0])
    print('P(diseaseY = 1 | age = 1)\t -> ', dis_a1)
    dis_a2 = len(np.where((vaccination['diseaseY'] == 1) & (vaccination['age'] == 2))[0]) / len(np.where(vaccination['age'] == 2)[0])
    print('P(diseaseY = 1 | age = 2)\t -> ', dis_a2)
    dis_a3 = len(np.where((vaccination['diseaseY'] == 1) & (vaccination['age'] == 3))[0]) / len(np.where(vaccination['age'] == 3)[0])
    print('P(diseaseY = 1 | age = 3)\t -> ', dis_a3)
    dis_a4 = len(np.where((vaccination['diseaseY'] == 1) & (vaccination['age'] == 4))[0]) / len(np.where(vaccination['age'] == 4)[0])
    print('P(diseaseY = 1 | age = 4)\t -> ', dis_a4)
    dis_a = [dis_a1, dis_a2, dis_a3, dis_a4]
    vacX_a1 = len(np.where((vaccination['vacX'] == 1) & (vaccination['age'] == 1))[0]) / len(np.where(vaccination['age'] == 1)[0])
    print('P(vacX = 1 | age = 1)\t -> ', vacX_a1)
    vacX_a2 = len(np.where((vaccination['vacX'] == 1) & (vaccination['age'] == 2))[0]) / len(np.where(vaccination['age'] == 2)[0])
    print('P(vacX = 1 | age = 2)\t -> ', vacX_a2)
    vacX_a3 = len(np.where((vaccination['vacX'] == 1) & (vaccination['age'] == 3))[0]) / len(np.where(vaccination['age'] == 3)[0])
    print('P(vacX = 1 | age = 3)\t -> ', vacX_a3)
    vacX_a4 = len(np.where((vaccination['vacX'] == 1) & (vaccination['age'] == 4))[0]) / len(np.where(vaccination['age'] == 4)[0])
    print('P(vacX = 1 | age = 4)\t -> ', vacX_a4)
    vacX_a = [vacX_a1, vacX_a2, vacX_a3, vacX_a4]
    print('P(knowsToRideABike = 1 | vacX = 0)\t -> ',
          len(np.where((vaccination['knowsToRideABike'] == 1) & (vaccination['vacX'] == 0))[0]) / len(np.where(vaccination['vacX'] == 0)[0]))
    print('P(knowsToRideABike = 1 | vacX = 1)\t -> ',
          len(np.where((vaccination['knowsToRideABike'] == 1) & (vaccination['vacX'] == 1))[0]) / len(np.where(vaccination['vacX'] == 1)[0]))
    
    fig = plt.figure()
    plt.title('diseaseY / age')
    plt.xlabel('age')
    plt.ylabel('disease score in %')
    plt.plot(dis_a)
    
    fig2 = plt.figure()
    plt.title('vaccination X / age')
    plt.xlabel('age')
    plt.ylabel('vaccination  score in %')
    plt.plot(vacX_a)
    # TODO What can you conclude from your results?

def main():
    aufg1a()
    marginal_p()
    preprocessing()
    conditional_p()
    plt.show()
    

if __name__ == "__main__":
  main()

