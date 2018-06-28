import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('./landmark-recog/google-landmarks-dataset/train.csv')
test_data = pd.read_csv('./landmark-recog/google-landmarks-dataset/test.csv')

print "Train data size", train_data.shape
print "Test data size", test_data.shape

# Occurance of landmark_id in decreasing order(Top categories)
temp = pd.DataFrame(train_data.landmark_id.value_counts().head(5))
print temp
temp.reset_index(inplace=True)
temp.columns = ['Landmark_ID','Occurence']

# Plot the most frequent landmark_ids
plt.figure(figsize = (7, 5))
plt.title('Landmark_ID Occurences')
sns.barplot(x="Landmark_ID", y="Occurence", data=temp,
            label="Occurence")

print temp
plt.show()










