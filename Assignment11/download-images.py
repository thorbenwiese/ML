# -*- coding: utf-8 -*-


import wget
import csv
import selectdata
from tabulate import tabulate

if __name__ == '__main__':
    f = 'google-landmarks-dataset/train.csv'
    # path = '/Users/wiese/Documents/UHH/Master/4.Semester/ML/Assignment11/selectedImages/'
    path = '/home/marcel/Dokumente/Uni/SOSE18/ML/Übung/u11/data/'
    count = 500
    with open(f) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        landmarks_occ = selectdata.getNmostIDs(f, 7)
        landmarks = list(map(int, [x[0] for x in landmarks_occ]))
        occ = list(map(int, [x[1] for x in landmarks_occ]))
        analysis = {}
        for l in landmarks:
            analysis[l] = 0
        for row in readCSV:
            try:
                index = int(row[2])
                #print(count, index, index in landmarks, count > 0)
                if index in landmarks and count > 0:
                    #print('valid index {}'.format(index))
                    try:
                        wget.download(row[1], path + str(row[0]) + '.jpg')
                        analysis[index] = analysis[index] +1
                        count -= 1
                    except:
                        print ('DOWNLOAD FAILED FOR ID:', str(row[0]))
            except:
                print ('NO LANDMARK ID')

    print ('number of rows in {}:'.format(f), len(open(f).readlines()))
    table = [(landmarks[i], occ[i], analysis[landmarks[i]]) for i in range(len(landmarks))]
    print (tabulate(table, headers=['landmark', 'occurence', 'downloaded']))
