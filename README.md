# pedestrian_retrieval


# Use generate_label.py to generate data label

unzip PedestrianRetrieval_vali in folder data/

run `generate_label.py` to generate train.csv, test.csv, eval.csv

in file train.csv, test.csv, eval.csv, each line express a triplet set file path, [reference image, positive image, negtive image]

# VGG_model
the implement of VGG network for pedestrian retrieval, detailed information see README.md under VGG_model

# compute metric: CMC
- with image features of gallery and probe, run `compute_distmat()` in cmc.py.
- with distmat, glabels and plabels obtained, run `count()` in cmc.py to get mean CMC.
- with distmat and array of gallery names obtained, run `sorted_image_names()` in cmc.py to get most similar images in gallery of each probe.
