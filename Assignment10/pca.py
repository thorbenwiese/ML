import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Quelle: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

number_of_components = 2 #? TODO plot for n > 2 ?

features = ['shape0','shape1','shape2','max','min','variance','numPixels'] 
target_feature = 'landmark_id' 
df = pd.read_csv('result.csv', header = 0)#names=features #TODO full-data-set

x = df.loc[:, features].values # Separating out the features
x_nan_indices = np.isnan(x)
x = x[~x_nan_indices.any(axis=1)] #remove rows where x contain nan
y = df.loc[:,[target_feature]].values # Separating out the target #TODO -> landmark_id
y = y[~x_nan_indices.any(axis=1)] #remove rows where x contain nan

x = StandardScaler().fit_transform(x) # Standardizing the features

pca = PCA(n_components=number_of_components)
principalComponents = pca.fit_transform(x)

column_gen = ['principal component {}'.format(i) for i in range(1, number_of_components + 1)]
principalDf = pd.DataFrame(data = principalComponents
             , columns = column_gen)

finalDf = pd.concat([principalDf, df[[target_feature]]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,2,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)  
ax.set_ylabel('Principal Component 2', fontsize = 15) 
ax.set_title('2 component PCA', fontsize = 20)

targets = np.unique(y)

#colors = ['r', 'g', 'b']
#colors = rand_color.generate(hue="blue", count=len(targets)) #list(np.random.choice(range(256), size=3))
'''
for target, color in zip(targets,colors):
    indicesToKeep = finalDf[target_feature] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
               '''
for target in targets:
    indicesToKeep = finalDf[target_feature] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               #, c = color
               ,cmap=plt.cm.get_cmap('RdBu')
               , s = 50)
#ax.legend(targets)
ax.grid()
print('pca.explained_variance_ratio', pca.explained_variance_ratio_)

ax = fig.add_subplot(1,2,2)
ax.set_xlabel('Principal Component 1', fontsize = 15)  
ax.set_ylabel('Principal Component 2', fontsize = 15) 
ax.set_title('2 component PCA', fontsize = 20)
targets = [2061, 60,51, 6599, 9633, 9779]
colors = ['b', 'g', 'r', 'c', 'm', 'y']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf[target_feature] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets, title="most frequent")
ax.grid()

plt.show()


