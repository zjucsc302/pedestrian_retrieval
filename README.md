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

# train note
1-6.20
checkpoint:model
model: vgg19, delete conv5,fc6,fc7,fc8, add fc6_new, fc7_new(100),  fine tuning conv4,fc6,fc7, resize image to 224*224, change VGG_MEAN
train: AdamOptimizer, initial_learning_rate = 0.001, self.learning_rate_decay_factor = 0.9
parameter: distance_alfa = 0.2
result: 110000 step, best 0.29

2-6.22
checkpoint:model_0.4
model: vgg19, delete conv5,fc6,fc7,fc8, add fc6_new, fc7_new(100),  fine tuning all, resize image to 224*224
train: AdamOptimizer, initial_learning_rate = 0.001, self.learning_rate_decay_factor = 0.96
parameter: distance_alfa = 0.2
result: 65000 step, best 0.2

3-6.23
checkpoint:model_224_112
model: vgg19, delete conv5,fc6,fc7,fc8, add fc6_new, fc7_new(100),  fine tuning conv4,fc6,fc7, resize image to 224*112
train: AdamOptimizer, initial_learning_rate = 0.001, self.learning_rate_decay_factor = 0.96
parameter: distance_alfa = 0.2
result: 230000 step, best 0.3

4-6.24
checkpoint:model_loss_split
model: vgg19, delete conv5,fc6,fc7,fc8, add fc6_new, fc7_new(100),  fine tuning conv4,fc6,fc7, resize image to 224*112, split loss in a batch
train: AdamOptimizer, initial_learning_rate = 0.001, self.learning_rate_decay_factor = 0.96
parameter: distance_alfa = 0.2
result: