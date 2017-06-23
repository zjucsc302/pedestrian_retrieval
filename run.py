from generate_label import *
from cmc import *
from VGG_model.train import *

if __name__ == '__main__':
    generate_path_label()
    train(retain_flag=False)
    # generate_features(predict_flag=True,gallery_flag=True)
    # generate_features(predict_flag=True,gallery_flag=False)
    # generate_features(predict_flag=False,gallery_flag=True) # train_1000_features
    # generate_features(predict_flag=False,gallery_flag=False)
    # train_1000_mAP()
    # valid_mAP()
    generate_predict_xml()