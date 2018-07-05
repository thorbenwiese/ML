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
import downloadImages
import shutil
from sklearn.cross_validation import train_test_split
import multiprocessing as mp
import time


#trainImgPath = '/Users/wiese/Documents/UHH/Master/4.Semester/ML/Assignment11/selectedImages/'
#testImgPath = '/Users/wiese/Documents/UHH/Master/4.Semester/ML/Assignment11/selectedImagesTest/'
trainImgPath = 'train/'
testImgPath = 'test/'
traincsv = 'google-landmarks-dataset/train.csv'
dataPath = 'data/'
number_of_img = 50000
thresh = number_of_img # thresh can be used to only select a couple of images -> reduce number-of_images for debugging
n_most_landmarks=20
testsize = 0.25




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
        print ('Error: ', e)
        return None

    return dsc

def prepareData():
    if not (os.path.exists(trainImgPath) or os.path.exists(testImgPath)):
        os.makedirs(trainImgPath)
        os.makedirs(testImgPath)
    if not (len(os.listdir(trainImgPath)) > 0 and len(os.listdir(testImgPath))):
        if not os.path.exists(dataPath):
            os.makedirs(dataPath)
        if not len(os.listdir(dataPath)) > 0:
            downloadImages.download(dataPath, traincsv, n_most_landmarks, number_of_img)
        data = os.listdir(dataPath)
        data = np.array(data)
        x_train, x_test = train_test_split(data, test_size=testsize)
        for x in x_train: shutil.move(os.path.join(dataPath, x), trainImgPath)
        for x in x_test: shutil.move(os.path.join(dataPath, x), testImgPath)

id_mapping = {}
with open(traincsv, mode='r') as infile:
  reader = csv.reader(infile)
  for rows in reader:
    key, value = rows[0], rows[2]
    id_mapping[key] = value

prepareData()

train_ids = [f.replace('.jpg','') for f in os.listdir(trainImgPath) if os.path.isfile(os.path.join(trainImgPath,     f))]
test_ids = [f.replace('.jpg','') for f in os.listdir(testImgPath) if os.path.isfile(os.path.join(testImgPath,     f))]



# Quelle multiprocessing: https://stackoverflow.com/questions/13446445/python-multiprocessing-safely-writing-to-a-file
def worker(imgId, q):
    print(imgId, 'started')
    start = time.clock()
    try:
        img = imageio.imread(trainImgPath + imgId + '.jpg')
        features = extract_features(img)
        q.put((imgId, [f for f in features], id_mapping[imgId]))
        success = 'success'
    except:
        success = 'failed'
    done = time.clock() - start
    res = 'Process for ImageID:' + str(imgId), success, 'done in: ',done#, 'counter: ', count
    print(res)
    return res

def listener(q):
    count = 0
    try:
        f = open("resultTRAIN.csv", 'w')
        wr = csv.writer(f)
        wr.writerow(['id', 'featureVector', 'landmark_id'])
        while 1:
            m = q.get()
            if m == 'kill':
                print('kill listener')
                break
            if count % 5 == 0:
                print(20*'=', count, ' / ', thresh, '  --> ',count/thresh, '% done ', 20*'=')
            count += 1
            wr.writerow(m)
    except:
        print('writer failed')

# must use Manager queue here, or will not work
manager = mp.Manager()
q = manager.Queue()
pool = mp.Pool(processes=4)

# put listener to work first
watcher = pool.apply_async(listener, (q,))

# fire off workers
jobs = []
for imgId in train_ids:
    if thresh > 0:
        thresh -= 1
        job = pool.apply_async(worker, (imgId, q))
        jobs.append(job)

# collect results from the workers through the pool result queue
for job in jobs:
    job.get()

# now we are done, kill the listener
q.put('kill')
pool.close()
pool.join()





'''
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
      print ('Run', count)
      count += 1
      try:
        img = imageio.imread(trainImgPath + imgId + '.jpg')
        features = extract_features(img)
        writer.writerow((imgId, [f for f in features], id_mapping[imgId]))
      except:
        print ('COULD NOT READ IMAGE WITH ID:', imgId)

# WRITE TEST RESULT
with open("resultTEST.csv", "w") as f:
  writer = csv.writer(f)
  for imgId in test_ids:
    if count == 0:
      writer.writerow(['id','featureVector'])
    if thresh > 0:
      thresh -= 1
      print ('Run', count)
      count += 1
      try:
        img = imageio.imread(testImgPath + imgId + '.jpg')
        features = extract_features(img)
        writer.writerow((imgId, [f for f in features]))
      except:
        print ('COULD NOT READ IMAGE WITH ID:', imgId)
'''

