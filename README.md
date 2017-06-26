# pedestrian_retrieval


# Use generate_label.py to generate data label

unzip PedestrianRetrieval_vali in folder data/

run `generate_label.py` to generate data/id_path.pkl, data/train_triplet_pair, data/train_1000_probe, data/train_1000_gallery, data/valid_probe.csv, data/valid_gallery.csv, data/predict_probe, data/predict_gallery.csv, data/predict_probe_name, data/predict_gallery_name.

id_path.pkl can speed up generate csv.

each line in csv epress:

train_triplet_pair: [ref image path, pos image path, neg image path, order].

train_1000_probe: [probe_train_1000_path, probe_train_1000_label, order].

train_1000_gallery: [gallery_train_path, gallery_train_label, order].

valid_probe.csv: [probe_valid_path, probe_valid_label, order].

valid_gallery.csv: [gallery_valid_path, gallery_valid_label, order].

predict_probe: [probe_predict_path, -1, order].

predict_gallery.csv: [gallery_predict_path, -1, order].

predict_probe_name: [probe_predict_name, order].

predict_gallery_name: [gallery_predict_name, order].

# Use VGG_model/train.py to train model

use train(retain_flag=True, start_step=1) to retrain the model.

if you stop the train, use train(retain_flag=False, start_step=the step to continue) to continue, model will be save every 5000 step.

after training, use 4 generate_features() to generate features for train and predict's gallery and probe.

# Use cmc.py to generate train mAP and data/predict_result.xml.

use train_1000_mAP() to generate train_1000 mAP.

use valid_mAP() to generate every 5000 steps' valid mAP.

use generate_predict_xml() to generate predict_result.xml, notice to follow the step under generate_predict_xml() to modify the xml by hand!!!

if you want to compute mAP after feature normalization, make sure normalize_flag=True, usually without normalization is better.
