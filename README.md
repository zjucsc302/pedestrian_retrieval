# pedestrian_retrieval


# Use generate_label.py to generate data label

unzip PedestrianRetrieval_vali in folder data/

run `generate_label.py` to generate train.csv, test.csv, eval.csv

in file train.csv, test.csv, eval.csv, each line express a triplet set file path, [reference image, positive image, negtive image]

# VGG_model
the implement of VGG network for pedestrian retrieval, detailed information see README.md under VGG_model

# generate gallery & probe
run `generaty_gallery()` in cmc.py to generate gallery, probe, along with their image labels.
(_Attention: gallery is generated from both train and val dataset. If only val data wanted, rewrite get_dict_ids_images() in generate_label.py_)

# compute metric: CMC
- with image features of gallery and probe, run `compute_distmat()` in cmc.py.
- with distmat, glabels and plabels obtained, run `count()` in cmc.py to get mean CMC.
