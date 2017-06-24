# for VGG: reshape every image to size(224, 224, 3)

import os, sys
from os import walk
import skimage.io
import numpy as np
import matplotlib.pyplot as plt


current_file_path = sys.path[0]
image_folder_path = os.path.join(current_file_path, 'data/PedestrianRetrieval_vali/vq_path')

f = []
for (dirpath, dirnames, filenames) in walk(image_folder_path):
    f.extend(filenames)
    break


def calc_dataset_mean(files_list):

    calc_number = 8000

    rgb = np.zeros([calc_number,3])
    shape = np.zeros([calc_number,2])

    for i in range(0,calc_number):
        image_path = os.path.join(current_file_path, 'data/PedestrianRetrieval_vali/vq_path', files_list[i])
        # print image_path
        each_rgb = skimage.io.imread(image_path)
        rgb[i, :] = np.mean(np.mean(each_rgb,0),0)
        shape[i,:] = each_rgb.shape[:2]
        # print np.mean(np.mean(each_rgb,0),0)

    print('mean rgb: ' + str(np.mean(rgb, 0)))
    plt.hist(rgb, bins=50)
    plt.show()
    print('mean shape: ' + str(np.mean(shape, 0)))
    plt.hist(shape, bins=50)
    plt.show()

if __name__ == '__main__':
    calc_dataset_mean(f)
