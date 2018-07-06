# -*- coding: utf-8 -*-


import wget
import csv
import os
from tabulate import tabulate


'''This script reads a csv which contain tupels of form (id, link, landmark) and  downloads a specified number (count) of images of the n most occurencing landmarks'''

def download(dataPath, traincsv ,n_most, count):
    if(not download_preconditions(dataPath, traincsv)):
        print('precondition error')
        return
    with open(traincsv) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        landmarks_occ = getNmostIDs(traincsv, n_most)
        landmarks_occ = [x for x in landmarks_occ if 'None' not in x]
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
                        wget.download(row[1], dataPath + str(row[0]) + '.jpg')
                        analysis[index] = analysis[index] +1
                        count -= 1
                    except:
                        print ('DOWNLOAD FAILED FOR ID:', str(row[0]))
            except:
                print ('NO LANDMARK ID')

    print ('number of rows in {}:'.format(traincsv), len(open(traincsv).readlines()))
    table = [(landmarks[i], occ[i], analysis[landmarks[i]]) for i in range(len(landmarks))]
    output_table = tabulate(table, headers=['landmark', 'occurence', 'downloaded'])
    print (output_table)
    open('download_statistik', 'w').write(output_table)

def getNmostIDs(file, n):
    print ('number of rows in {}:'.format(file), len(open(file).readlines()))
    with open(file, mode='r') as infile:
        reader = csv.reader(infile)
        mydict = {}
        for rows in reader:
            key, value = rows[2], (rows[0], rows[1])
            if key not in mydict:
                mydict[key] = [value]
            elif type(mydict[key]) == list:
                mydict.get(key).append(value)
            else:
                print('type error')

    sorted_x = sorted(mydict.items(), key=lambda kv: len(kv[1]), reverse=True)

    selectedIds = []

    for i in range(n):
        print ('landmark_id', sorted_x[i][0], 'occurs', len(sorted_x[i][1]), ' times')
        selectedIds.append((sorted_x[i][0], len(sorted_x[i][1])))

    return selectedIds

def download_preconditions(dataPath, traincsv):
    if os.path.exists(dataPath):
        return os.path.isdir(dataPath) and os.path.exists(traincsv)
    else:
        os.makedirs(dataPath)
        return os.path.exists(traincsv)
