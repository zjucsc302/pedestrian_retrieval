# train note

recommend use typore to view this note

| date | checkpoint        | model                                    | train                                    | loss                                     | result                                | conclusion                               |
| ---- | ----------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ------------------------------------- | ---------------------------------------- |
| 7.10 | resnet            | resnet50, delete fc, add fc1(1024), bn, relu, fc2(128), fine tuning fc1, bn, fc2, no dropout, crop image, resize image to 224*112, data enhancement | AdamOptimizer, initial_learning_rate = 0.0001, ecay_rate = 0.1, decay_steps = 10000, return_id_num = 20 | batch hard triplet loss, soft-margin     | step 20000, mAP 0.52                  | baseline                                 |
| 7.11 | resnet_softmax    | resnet50, delete fc, add fc1(1024), bn, relu, fc2(128), **softmax**, fine tuning fc1, bn, fc2, no dropout, crop image, resize image to 224*112, data enhancement | AdamOptimizer, initial_learning_rate = 0.0001, **decay_rate = 0.5**, decay_steps = 10000, return_id_num = 20 | batch hard triplet loss, soft-margin     | output festures trend to 0            | softmax is worse                         |
| 7.11 | resnet_no_pool    | resnet50, delete **global pool**, fc, add fc1(1024), bn, relu, fc2(128), fine tuning fc1, bn, fc2, no dropout, crop image, resize image to 224*112, data enhancement | AdamOptimizer, initial_learning_rate = 0.0001, decay_rate = 0.5, decay_steps = 10000, return_id_num = 20 | batch hard triplet loss, soft-margin,    | output festures trend to 0            | delete global pool is worse              |
| 7.11 | resnet_finetuning | resnet50, fc, add fc1(1024), bn, relu, fc2(128), fine tuning **resnet50**, fc1, bn, fc2, no dropout, crop image, resize image to 224*112, data enhancement | AdamOptimizer, initial_learning_rate = 0.0001, decay_rate = 0.5, decay_steps = 10000, return_id_num = 20 | batch hard triplet loss, soft-margin     | step 15000, mAP 0.88                  | fine turning resnet is pefect            |
| 7.11 | resnet_finetuning | resnet50, fc, add fc1(1024), bn, relu, fc2(128), fine tuning resnet50, fc1, bn, fc2, no dropout, crop image, resize image to 224*112, data enhancement | **train after resnet_finetuning step 20000**, AdamOptimizer, **initial_learning_rate = 0.000001**, decay_rate = 0.5, decay_steps = 10000, return_id_num = 20 | **batch hardest triplet loss**, soft-margin | step 25000, mAP 0.89                  | batch hardest triplet loss seems no much help |
| 7.11 | resnet_noprecess  | resnet50, fc, add fc1(1024), bn, relu, fc2(128), fine tuning resnet50, fc1, bn, fc2, no dropout, crop image, resize image to 224*112, **no data enhancement** | AdamOptimizer, initial_learning_rate = 0.0001, decay_rate = 0.5, decay_steps = 10000, return_id_num = 20 | batch hard triplet loss, soft-margin     | step 30000, mAP 0.92 (train 0.97)     | no data enhancement seems ok, maybe train data set is big enough |
| 7.12 | resnet_enhance    | resnet50, fc, add fc1(1024), bn, relu, fc2(128), fine tuning resnet50, fc1, bn, fc2, no dropout, crop image, resize image to 224*112, **update data enhancement** | AdamOptimizer, initial_learning_rate = 0.0001, decay_rate = 0.5, decay_steps = 10000, return_id_num = 20 | batch hard triplet loss, soft-margin     | **step 25000, mAP 0.92 (train 0.98)** | update data enhancement seams good       |




















