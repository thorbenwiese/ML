# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy
from scipy import misc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# read CSV file and return columns
def readCSVFile(f, train):

  ids = []
  urls = []
  landmark_ids = []
  with open(f) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
      ids.append(row[0])
      urls.append(row[1])
      if train == True:
        landmark_ids.append(row[2])

  if train == True:
    return ids[:100], urls[:100], landmark_ids[:100]

  return ids[:100], urls[:100]



# use PILLOW to read images from url
# Image.open(urllib2.urlopen(url))
# Image.open(requests.get(url, stream=True).raw)



def getStatistics(img):

  print img.shape
  print img.dtype
  print img.max(), img.min()
  # mean, median, variances, empirical probabilities
  # imbalances, non-normalized features





def extractFeatures(img):
  # extract features
  # are they correlated
  # cluster features with respect to correlation coefficient
  # which features are important
  print 'Features extracted.'


def preprocessData(img):
  # preprocess data
  # cluster data (or explain why not possible) (maybe cluster only some features)
  print 'Data preprocessed.'


def dimReduction(img):
  # dimensionality reduction method (e.g. PCA) or CNN Feature Map?
  print 'Dimensions Reduced.'





def main():

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



  print('Reading files...')
  #index_ids, index_urls = readCSVFile("landmark-recog/google-landmarks-dataset/index.csv", False)
  #train_ids, train_urls, train_landmark_ids = readCSVFile("landmark-recog/google-landmarks-dataset/train.csv", True)
  #test_ids, test_urls = readCSVFile("landmark-recog/google-landmarks-dataset/test.csv", False)
  print('Files Read.')
  
  imgId = 'fdbac32ad9bbf6e5'
  try:
    img = misc.imread('/Volumens/WIESE/landmark-images/train/' + imgId + '.jpg')
    x = getStatistics(img)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()
  except:
    print 'Image with ID:', imgId, 'could not be read!'


'''
We selected the distribution of landmarks, the dimenions of the images, the amount of pixels, the pixel range (from min to max) and the pixel variance (max - min) to get a better understanding of what the images' shapes are.
'''


# execute main
if __name__ == "__main__":
  main()
  plt.show()
