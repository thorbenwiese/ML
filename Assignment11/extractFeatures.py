# import the necessary packages
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import imageio
import csv
import downloadImages
import shutil
from sklearn.cross_validation import train_test_split
import multiprocessing as mp
import time

'''
This script extracts feature vectors for a given train- and test-set (folders) by using multiprocessing (for reasons of
performance). The feature vectors are stored in csv-files. If there don't exist sets, this script starts the
downloadImage-script an split the result into train- and test-set

Caution! Actually the feature-extraction for test-sets seems not to work, please use extractFeaturesTest.py for this.
'''


#trainImgPath = '/Users/wiese/Documents/UHH/Master/4.Semester/ML/Assignment11/selectedImages/'
#testImgPath = '/Users/wiese/Documents/UHH/Master/4.Semester/ML/Assignment11/selectedImagesTest/'
trainImgPath = 'train/'
testImgPath = 'test/'
traincsv = 'google-landmarks-dataset/train.csv'
dataPath = 'data/'
number_of_img = 200
thresh = number_of_img # thresh can be used to only select a couple of images -> reduce number-of_images for debugging
n_most_landmarks=7
testsize = 0.25
count = 0 # ignore this -> just a counter for monitoring progress

g_start = time.time()




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
print('time for data preparation: ', time.time() - g_start)

train_ids = [f.replace('.jpg','') for f in os.listdir(trainImgPath) if os.path.isfile(os.path.join(trainImgPath,     f))]
test_ids = [f.replace('.jpg','') for f in os.listdir(testImgPath) if os.path.isfile(os.path.join(testImgPath,     f))]



# Quelle multiprocessing: https://stackoverflow.com/questions/13446445/python-multiprocessing-safely-writing-to-a-file
def worker(imgId, path, q):
    print(imgId, 'started')
    start = time.clock()
    try:
        img = imageio.imread(path + imgId + '.jpg')
        features = extract_features(img)
        q.put((imgId, [f for f in features], id_mapping[imgId]))
        success = 'success'
    except:
        success = 'failed'
    done = time.clock() - start
    res = 'Process for ImageID:' + str(imgId), success, 'done in: ',done#, 'counter: ', count
    print(res)
    return res

def listener(l_csv, q):
    global count
    try:
        f = open(l_csv, 'w')
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


#create feature vectors for trai
# must use Manager queue here, or will not work
manager = mp.Manager()
q = manager.Queue()
pool = mp.Pool(processes=4)

# put listener to work first
watcher = pool.apply_async(listener, ("resultTRAIN.csv", q,))

# fire off workers
jobs = []
for imgId in train_ids:
    if thresh > 0:
        thresh -= 1
        job = pool.apply_async(worker, (imgId, trainImgPath, q))
        jobs.append(job)

# collect results from the workers through the pool result queue
for job in jobs:
    job.get()

# now we are done, kill the listener
q.put('kill')
pool.close()
pool.join()

time.sleep(5)

#create feature vectors for test
# must use Manager queue here, or will not work
manager = mp.Manager()
q = manager.Queue()
pool = mp.Pool(processes=4)

# put listener to work first
watcher = pool.apply_async(listener, ("resultTEST.csv", q,))

# fire off workers
jobs = []
for imgId in test_ids:
    if thresh > 0:
        thresh -= 1
        job = pool.apply_async(worker, (imgId, testImgPath, q))
        jobs.append(job)

# collect results from the workers through the pool result queue
for job in jobs:
    job.get()

# now we are done, kill the listener
q.put('kill')
pool.close()
pool.join()

print('time over all: ', time.time() - g_start)
