# for VGG: reshape every image to size(224, 224, 3)


import os, sys

from os import walk
import skimage.io
import numpy as np

current_file_path = sys.path[0]
image_folder_path = os.path.join(current_file_path, 'data/PedestrianRetrieval_vali/vr_path')

f = []
for (dirpath, dirnames, filenames) in walk(image_folder_path):
    f.extend(filenames)
    break


def calc_dataset_mean(files_list):

    calc_number = 1000

    rgb_mean = np.zeros([calc_number,3])

    for i in range(0,calc_number):
        image_path = os.path.join(current_file_path, 'data/PedestrianRetrieval_vali/vr_path', files_list[i])
        print image_path

        each_rgb = skimage.io.imread(image_path)
        rgb_mean[i, :] = np.mean(np.mean(each_rgb,0),0)
        print np.mean(np.mean(each_rgb,0),0)

    print np.mean(rgb_mean, 0)


#calc_dataset_mean(f)

each_rgb = skimage.io.imread('images2.jpg')
print each_rgb.shape
print np.mean(np.mean(each_rgb,0),0)

