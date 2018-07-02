import wget
import csv

if __name__ == '__main__':

    f = 'google-landmarks-dataset/train.csv'
    selectedIds = [2061, 6051, 6599, 9633, 9779, 60, 51]
    count = 500

    with open(f) as csvfile:
      readCSV = csv.reader(csvfile, delimiter=',')
      skip = 0
      for row in readCSV:
        try:
          if skip != 0 and int(row[2]) in selectedIds and count > 0:
            count -= 1
            try:
              wget.download(row[1], '/Users/wiese/Documents/UHH/Master/4.Semester/ML/Assignment11/selectedImages/' + str(row[0]) + '.jpg') 
            except:
              print 'DOWNLOAD FAILED FOR ID:', str(row[0])
          skip = 42
        except:
          print 'NO LANDMARK ID'
