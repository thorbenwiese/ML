import wget
import csv

if __name__ == '__main__':

    f = 'landmark-recog/google-landmarks-dataset/train.csv'
    urls = []
    ids = []
    with open(f) as csvfile:
      readCSV = csv.reader(csvfile, delimiter=',')
      skip = 0
      for row in readCSV:
        if skip != 0:
          try:
            wget.download(row[1], '/Volumes/WIESE/landmark-images/train/' + str(row[0]) + '.jpg') 
          except:
            print 'DOWNLOAD FAILED FOR ID:', str(row[0])
        skip = 42
