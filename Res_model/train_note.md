# train note

recommend use typore to view this note

| date | checkpoint | model                                    | train                                    | loss                                  | result | conclusion |
| ---- | ---------- | ---------------------------------------- | ---------------------------------------- | ------------------------------------- | ------ | ---------- |
| 7.10 | resnet     | resnet50, delete fc, add fc1(1024), bn, relu, fc2(128), fine tuning fc1, bn, fc2, no dropout, resize image to 224*112 | AdamOptimizer, initial_learning_rate = 0.0001, ecay_rate = 0.1, decay_steps = 10000, return_id_num = 20 | batch hard triplet loss, soft-margin, |        |            |

