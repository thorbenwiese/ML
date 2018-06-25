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
          wget.download(row[1], '/Users/wiese/Downloads/images/' + str(row[0]) + '.jpg') 
        skip = 42
