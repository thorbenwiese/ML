import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Quelle: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

number_of_components = 2 #? TODO plot for n > 2 ?

features = ['shape0','shape1','shape2','max','min','variance','numPixels']#,'landmark_id'] #TODO
df = pd.read_csv('result_without_landmarks.csv', header = 0)#names=features) #TODO

x = df.loc[:, features].values # Separating out the features
x_nan_indices = np.isnan(x)
x = x[~x_nan_indices.any(axis=1)] #remove rows where x contain nan
y = df.loc[:,['numPixels']].values # Separating out the target #TODO -> landmark_id
y = y[~x_nan_indices.any(axis=1)] #remove rows where x contain nan

x = StandardScaler().fit_transform(x) # Standardizing the features

pca = PCA(n_components=number_of_components)
principalComponents = pca.fit_transform(x)

column_gen = ['principal component'+i for i in range(number_of_components)]
principalDf = pd.DataFrame(data = principalComponents
             , columns = column_gen)

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

# TODO pca.explained_variance_ratio_