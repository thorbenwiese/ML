from PIL import Image
from matplotlib import pyplot as plt
import glob, os
import pprint
import json
import random


######################## 3b ########################
'''
There are 5749 classes (persons) with 13233 images. 
The distribution of these images is not balanced. For each person there are up to 530 images (George Bush),
but for most of them only 1-20 images exists.  
'''

def humansize(nbytes):
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])

def analyse(print2console, print2file):
    dataset_path = './data/lfw_home/lfw_funneled'
    class_counter = 0
    image_counter = 0
    dataset_meta = {}
    for classname in os.listdir(dataset_path):
    #http://effbot.org/imagingbook/image.htm
        class_path = dataset_path+'/'+classname
        if os.path.isdir(class_path):
            class_counter += 1
            meta = {'images':[]}
            class_imagecounter = 0
            for image in os.listdir(class_path):
                image_path = class_path+'/'+image
                if image.endswith('.jpg'):
                    im = Image.open(image_path)
                    width, height = im.size
                    #print(image, '->', image_path, width, height, im.getextrema(), im.mode, humansize(os.path.getsize(image_path)))
                    meta_image = {
                            'class': os.path.splitext(image)[0],
                            'width': width,
                            'height': height,
                            'pixel_range': im.getextrema(),
                            'mode': im.mode,
                            'size': humansize(os.path.getsize(image_path)),
                            'path': image_path
                        }
                    meta['images'].append(meta_image)
                    image_counter += 1
                    class_imagecounter += 1
                meta['class_image_counter'] = class_imagecounter
            dataset_meta[classname] = meta
    dataset_meta['class_counter']= class_counter
    dataset_meta['image_counter'] = image_counter
    if print2console:
        pprint.pprint(dataset_meta, width=1)
    if print2file:
        with open('result3b.json', 'w') as file:
            file.write(json.dumps(dataset_meta))
    return dataset_meta
     
def random_images(set, i, sample_size):
    choices = random.sample(list(set), i)
    print('choosen (name, image count, image count >= sample_size):', 
          [(x, len(set[x]['images']), len(set[x]['images']) >= sample_size) for x in choices])
    for choice in choices:
        sample_size_temp = sample_size
        fig = plt.figure()
        while True:
            # try-except da für einige Personen nur 1 Bild und 1 < sample_size
            try:
                sample = random.sample(set[choice]['images'], sample_size_temp)
                print(choice)
                pprint.pprint(sample, width=1)
                print('\n\n')
                images = [s['path'] for s in sample]
                for i in range(len(images)):
                    x = fig.add_subplot(1,sample_size_temp,i+1)
                    x.set_title(sample[i]['class'])
                    plt.imshow(Image.open(images[i]))
                fig.canvas.set_window_title(choice)
            except ValueError:
                sample_size_temp -= 1
                continue
            break
    plt.show()
            
        
    


meta = analyse(False, True)
random_images(meta, 5, 3)




