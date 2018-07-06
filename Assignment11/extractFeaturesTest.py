# import the necessary packages
import numpy as np
import cv2
import os
import imageio
import csv
import multiprocessing as mp
import time


'''
Pricipially this script does the same as extractFeatures.py (extract feature-vectors to csv) with the difference that
 the feature extraction for train-sets and the data-preparation were removed. As soon as the broken feature extraction
 for test-sets (in extractFeatures.py) is fixed this script may be removed.
'''


#trainImgPath = 'train/'
testImgPath = 'test/'
traincsv = 'google-landmarks-dataset/train.csv'
dataPath = 'data/'
thresh = len(os.listdir(testImgPath)) # In difference to extractFeature.py in this script thresh isn't a threshold but the number of all images in testimage-folder
count = 0 # ignore this -> just a counter for monitoring progress

g_start = time.time()


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

id_mapping = {}
with open(traincsv, mode='r') as infile:
  reader = csv.reader(infile)
  for rows in reader:
    key, value = rows[0], rows[2]
    id_mapping[key] = value

#train_ids = [f.replace('.jpg','') for f in os.listdir(trainImgPath) if os.path.isfile(os.path.join(trainImgPath,     f))]
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
                print(20*'=', count, ' / ', thresh, '  --> ',count*100/thresh, '% done ', 20*'=')
            if count % 10 == 0:
                print('time since start: ', time.time() - g_start)
            count += 1
            wr.writerow(m)
    except:
        print('writer failed')


#create feature vectors for test'
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