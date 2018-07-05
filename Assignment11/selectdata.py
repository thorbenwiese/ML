import csv


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
