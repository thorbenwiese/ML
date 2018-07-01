import re
import csv
import pandas as pd

p = "result.txt"
result = [['id','shape0','shape1','shape2', 'max', 'min', 'variance', 'numPixels', 'landmark_id']]
train_data = pd.read_csv('./landmark-recog/google-landmarks-dataset/train.csv')

with open(p, "r") as fp:
    for i in fp.readlines():
        tmp = re.sub("\(|\)|\'|.jpg|\n", "", i).split(', ')
        tmp[1:8] = [int(x) for x in tmp[1:8]]
        try:
            result.append(tmp)
            #result.append((eval(tmp[0]), eval(tmp[1])))
        except:pass

with open("result_without_landmarks.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(result)