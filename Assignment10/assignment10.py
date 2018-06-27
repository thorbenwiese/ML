# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy
from scipy import misc
import matplotlib.pyplot as plt

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


















def main():

  print('Reading files...')
  #index_ids, index_urls = readCSVFile("landmark-recog/google-landmarks-dataset/index.csv", False)
  #train_ids, train_urls, train_landmark_ids = readCSVFile("landmark-recog/google-landmarks-dataset/train.csv", True)
  #test_ids, test_urls = readCSVFile("landmark-recog/google-landmarks-dataset/test.csv", False)
  print('Files Read.')
  
  img = misc.imread('/Users/wiese/Downloads/images/fdbac32ad9bbf6e5.jpg')
  print img.shape
  print img.dtype
  print img.max(), img.min()
  plt.imshow(img, cmap=plt.cm.gray)
  plt.show()



# execute main
if __name__ == "__main__":
  main()
