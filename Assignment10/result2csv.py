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
        q = "id == '{}'".format(tmp[0])
        try:
            tmp.append(int(train_data.query(q).landmark_id.values[0])) #TODO id sollte eindeutig sein und eindeutige landmark
        except:
            tmp.append('None')
        try:
            result.append(tmp)
            print('landmark for ', tmp[0] , ' is ',tmp[8])
        except:pass

with open("result.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(result)