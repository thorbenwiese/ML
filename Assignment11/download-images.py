# -*- coding: utf-8 -*-

import wget
import csv
import selectdata

if __name__ == '__main__':
    f = 'google-landmarks-dataset/train.csv'
    #path = '/Users/wiese/Documents/UHH/Master/4.Semester/ML/Assignment11/selectedImages/'
    path = '/home/marcel/Dokumente/Uni/SOSE18/ML/Übung/u11/data/'
    count = 500
    with open(f) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        landmarks = map(int,selectdata.getNmostIDs(f, 7))
        for row in readCSV:
            try:
                index = int(row[2])
                #print(count, skip, index, b)
                if index in landmarks and count > 0:
                    print('valid index {}'.format(index))
                    try:
                        wget.download(row[1], path + str(row[0]) + '.jpg')
                        count -= 1
                    except:
                        print ('DOWNLOAD FAILED FOR ID:', str(row[0]))
            except:
                print ('NO LANDMARK ID')

    print ('number of rows in {}:'.format(f), len(open(f).readlines()))
