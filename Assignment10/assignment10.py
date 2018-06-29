# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy
import imageio
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import progressbar as pb
import os

#define progress timer class
class progress_timer:

    def __init__(self, n_iter, description="Something"):
        self.n_iter         = n_iter
        self.iter           = 0
        self.description    = description + ': '
        self.timer          = None
        self.initialize()

    def initialize(self):
        #initialize timer
        widgets = [self.description, pb.Percentage(), ' ',   
                   pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
        self.timer = pb.ProgressBar(widgets=widgets, maxval=self.n_iter).start()

    def update(self, q=1):
        #update timer
        self.timer.update(self.iter)
        self.iter += q

    def finish(self):
        #end timer
        self.timer.finish()

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
    return ids, urls, landmark_ids

  return ids, urls


def main():

  train_data = pd.read_csv('./landmark-recog/google-landmarks-dataset/train.csv')
  test_data = pd.read_csv('./landmark-recog/google-landmarks-dataset/test.csv')

  print "Train data size", train_data.shape
  print "Test data size", test_data.shape
  print ''

  c = train_data.landmark_id.value_counts()

  temp = pd.DataFrame(train_data.landmark_id.value_counts().head(100))
  temp.reset_index(inplace=True)
  temp.columns = ['Landmark_ID','Occurence']

  # Plot the most frequent landmark_ids
  plt.figure(figsize = (7, 5))
  sns.barplot(x="Landmark_ID", y="Occurence", data=temp,
            label="Occurence")

  # Occurance of landmark_id in decreasing order(Top categories)
  temp = pd.DataFrame(train_data.landmark_id.value_counts().head(5))
  temp.reset_index(inplace=True)
  temp.columns = ['Landmark_ID','Occurence']
  print temp
  print ''

  # Plot the most frequent landmark_ids
  plt.figure(figsize = (7, 5))
  plt.title('Landmark_ID Occurences')
  sns.barplot(x="Landmark_ID", y="Occurence", data=temp,
            label="Occurence")


  imgPath = '/Volumes/WIESE/landmark-images/train/'

  print('Reading files...\n')
  #index_ids, index_urls = readCSVFile("landmark-recog/google-landmarks-dataset/index.csv", False)
  #train_ids, train_urls, train_landmark_ids = readCSVFile("landmark-recog/google-landmarks-dataset/train.csv", True)
  #test_ids, test_urls = readCSVFile("landmark-recog/google-landmarks-dataset/test.csv", False)
  train_ids = [f for f in os.listdir(imgPath) if os.path.isfile(os.path.join(imgPath, f))]
  print('Files Read.\n')

  print 'AMOUNT IMAGES:', len(train_ids)
  print ''
  
  pt = progress_timer(description='Analyzing Images', n_iter=len(train_ids))
  failed_count = 0

  maxVals = []
  minVals = []
  variances = []
  numPixels = []

  full_data = []

  for imgId in train_ids:
    pt.update()
    if imgId != 'id':
      try:

        img = imageio.imread(imgPath + imgId)

        maxVals.append(img.max())
        minVals.append(img.min())
        variances.append(img.max() - img.min())
        numPixels.append(img.shape[0] * img.shape[1])

        full_data.append((imgId, img.shape, img.max(), img.min(), img.max() - img.min(), img.shape[0] * img.shape[1]))
      except:
        failed_count *= 1

  pt.finish()
  print '\n# Images failed:\n', failed_count

  #os.system('echo "Data" > output.txt')
  for d in full_data:
    print d
    #os.system("echo " + d + " >> output.txt")

  print '-'*40
  print 'maxVals'
  print 'Mean:', np.mean(maxVals)
  print 'Max:', np.max(maxVals)
  print 'Min:', np.min(maxVals)
  print 'Median:', np.median(maxVals)
  print '-'*40
  print ''
  print '-'*40
  print 'minVals'
  print 'Mean:', np.mean(minVals)
  print 'Max:', np.max(minVals)
  print 'Min:', np.min(minVals)
  print 'Median:', np.median(minVals)
  print '-'*40
  print ''
  print '-'*40
  print 'variances'
  print 'Mean:', np.mean(variances)
  print 'Max:', np.max(variances)
  print 'Min:', np.min(variances)
  print 'Median:', np.median(variances)
  print '-'*40
  print ''
  print '-'*40
  print 'numPixels'
  print 'Mean:', np.mean(numPixels)
  print 'Max:', np.max(numPixels)
  print 'Min:', np.min(numPixels)
  print 'Median:', np.median(numPixels)
  print '-'*40
  print ''

  plt.figure()
  plt.scatter(maxVals, range(len(maxVals)))
  plt.title('MaxVals')

  plt.figure()
  plt.scatter(minVals, range(len(minVals)))
  plt.title('MinVals')

  plt.figure()
  plt.scatter(variances, range(len(variances)))
  plt.title('Variances')

  plt.figure()
  plt.scatter(numPixels, range(len(numPixels)))
  plt.title('numPixels')
  plt.figure()
  plt.hist(numPixels, bins='auto')
  plt.title('numPixels Hist')


# execute main
if __name__ == "__main__":
  start = time.time()
  main()
  print 'Calculation took:', time.time() - start, 'seconds.'
  plt.show()

