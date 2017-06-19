import numpy as np
import csv

from cmc import count,compute_distmat

features = np.load('VGG_model/result/test_features/test_features_step-30000.npy')

features_id = []

with open('test.csv') as f:
  reader = csv.reader(f)
  for row in reader:
    features_id.append(row[1])


gallery = []
with open('gallery.csv') as f:
  reader = csv.reader(f)
  for row in reader:
      gallery.append(row)
gallery_num = len(gallery)

features_gallery = features[0:gallery_num, :]
features_probe = features[gallery_num:, :]

id_gallery = features_id[0:gallery_num]
id_probe = features_id[gallery_num:]

distmat = compute_distmat(features_gallery,features_probe)
cmc_mean = count(distmat=distmat,glabels=np.array(id_gallery),plabels=np.array(id_probe),n_selected_labels=500,n_repeat=10)

print np.mean(cmc_mean)


