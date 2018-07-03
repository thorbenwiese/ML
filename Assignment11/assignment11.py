# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import imageio
import csv

'''
def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()
'''

# Feature extractor
def extract_features(image, vector_size=32):
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them. 
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print 'Error: ', e
        return None

    return dsc

imgPath = '/Users/wiese/Documents/UHH/Master/4.Semester/ML/Assignment11/selectedImages/'

train_ids = [f.replace('.jpg','') for f in os.listdir(imgPath) if os.path.isfile(os.path.join(imgPath,     f))]

imgVectors = [['id','featureVector']]
count = 0

for imgId in train_ids:
  print 'Run', count
  count += 1
  try:
    img = imageio.imread(imgPath + imgId + '.jpg')
    imgVectors.append([imgId, extract_features(img)])
  except:
    print 'COULD NOT READ IMAGE WITH ID:', imgId

with open("result.csv", "w") as f:
  writer = csv.writer(f)
  writer.writerows(imgVectors)
