import numpy as np
import cv2
import os
import imageio
import csv

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

id_mapping = {}
with open('google-landmarks-dataset/train.csv', mode='r') as infile:
  reader = csv.reader(infile)
  for rows in reader:
    key, value = rows[0], rows[2]
    id_mapping[key] = value

imgPath = '/Users/wiese/Documents/UHH/Master/4.Semester/ML/Assignment11/selectedImages/'

testImgPath = '/Users/wiese/Documents/UHH/Master/4.Semester/ML/Assignment11/selectedImagesTest/'

train_ids = [f.replace('.jpg','') for f in os.listdir(imgPath) if os.path.isfile(os.path.join(imgPath,     f))]

test_ids = [f.replace('.jpg','') for f in os.listdir(testImgPath) if os.path.isfile(os.path.join(testImgPath,     f))]

count = 0

# thresh can be used to only select a couple of images
thresh = 500

# WRITE TRAIN RESULT
with open("resultTRAIN.csv", "w") as f:
  writer = csv.writer(f)
  for imgId in train_ids:
    if count == 0:
      writer.writerow(['id','featureVector','landmark_id'])
    if thresh > 0:
      thresh -= 1
      print 'Run', count
      count += 1
      try:
        img = imageio.imread(imgPath + imgId + '.jpg')
        features = extract_features(img)
        writer.writerow((imgId, [f for f in features], id_mapping[imgId]))
      except:
        print 'COULD NOT READ IMAGE WITH ID:', imgId

# WRITE TEST RESULT
with open("resultTEST.csv", "w") as f:
  writer = csv.writer(f)
  for imgId in test_ids:
    if count == 0:
      writer.writerow(['id','featureVector'])
    if thresh > 0:
      thresh -= 1
      print 'Run', count
      count += 1
      try:
        img = imageio.imread(testImgPath + imgId + '.jpg')
        features = extract_features(img)
        writer.writerow((imgId, [f for f in features]))
      except:
        print 'COULD NOT READ IMAGE WITH ID:', imgId

