# train note
1-6.20
checkpoint:model
model: vgg19, delete conv5,fc6,fc7,fc8, add fc6_new, fc7_new(100),  fine tuning conv4,fc6,fc7, resize image to 224*224, change VGG_MEAN, normalize output
train: AdamOptimizer, initial_learning_rate = 0.001, learning_rate_decay_factor = 0.9
loss: base triplet, distance_alfa = 0.2
result: 110000 step, best mAP 0.29
conclusion: base triplet is ok

2-6.22
checkpoint:model_0.4
model: vgg19, delete conv5,fc6,fc7,fc8, add fc6_new, fc7_new(100),  fine tuning all, resize image to 224*224, normalize output
train: AdamOptimizer, initial_learning_rate = 0.001, learning_rate_decay_factor = 0.96
loss: base triplet, distance_alfa = 0.2
result: 65000 step, best mAP 0.2
conclusion: fine turning all layers is slow

3-6.23
checkpoint:model_224_112
model: vgg19, delete conv5,fc6,fc7,fc8, add fc6_new, fc7_new(100),  fine tuning conv4,fc6,fc7, resize image to 224*112, normalize output
train: AdamOptimizer, initial_learning_rate = 0.001, learning_rate_decay_factor = 0.96
loss: base triplet, distance_alfa = 0.2
result: 230000 step, best mAP 0.3
conclusion: decrease cnn input size can not improve

4-6.24
checkpoint:model_224_112
model: vgg19, delete conv5,fc6,fc7,fc8, add fc6_new, fc7_new(100),  fine tuning conv4,fc6,fc7, resize image to 224*112, split loss in a batch, normalize output
train: AdamOptimizer, initial_learning_rate = 0.001, learning_rate_decay_factor = 0.96
loss: base triplet, distance_alfa = 0.2
result: 350000 step, best mAP 0.3
conclusion: SGD is not better than min-batch

4-6.25
checkpoint:model_trip_improve
model: vgg19, delete conv5,fc6,fc7,fc8, add fc6_new, fc7_new(100),  fine tuning conv4,fc6,fc7, resize image to 224*112, normalize output
train: AdamOptimizer, initial_learning_rate = 0.001, learning_rate_decay_factor = 0.9
loss: improved triplet, tau1 = 0.4, tau2 = 0.01, beta = 0.002
result: 225000 step, best mAP 0.24 (when tau1 = 1.0, 30000 step, best mAP 0.1, mAP unstable)
conclusion: tau1 = 0.2 better than 0.4 better than 1.0

4-6.26
checkpoint:model_trip_improve
model: vgg19, delete conv5,fc6,fc7,fc8, add fc6_new, fc7_new(100),  fine tuning conv4,fc6,fc7, resize image to 224*112, normalize output
train: AdamOptimizer, initial_learning_rate = 0.00001, learning_rate_decay_factor = 0.9
loss: improved triplet, tau1 = 0.4, tau2 = 0.01, beta = 0.002
result: 35000 step, mAP 0.4, seams can improve with more step(train with normalize feature, compute mAP and predict without normalize)
conclusion: 1. initial_learning_rate = 0.00001 is better than 0.01, when 0.01, features will be unconvergence and reach 10^10. 2. loss and net output can be used to select learning rate.

checkpoint:model_trip_improve
model: vgg19, delete conv5,fc6,fc7,fc8, add fc6_new, fc7_new(100),  fine tuning conv4,fc6,fc7, resize image to 224*112
train: AdamOptimizer, initial_learning_rate = 0.0001, learning_rate_decay_factor = 0.9
loss: improved triplet, tau1 = 0.2,0.4,1.0, tau2 = 0.01, beta = 0.002,0.02.0.2
result: 1. when initial_learning_rate = 0.00001, 5000 step, mAP 0.24, 10000 step, mAP 0.28. 2.when initial_learning_rate = 0.0001, 5000 step, mAP 0.05, 10000 step, mAP 0.07.
conclusion: without normalize, all features will tend to 0 to keep loss

checkpoint:model_trip_improve_1
model: vgg19, delete conv5,fc6,fc7,fc8, add fc6_new, fc7_new(100),  fine tuning conv4,fc6,fc7, resize image to 224*112, normalize output
train: AdamOptimizer, initial_learning_rate = 0.00001, learning_rate_decay_factor = 0.9
loss: improved triplet, tau1 = 1.0, tau2 = 0.01, beta = 0.002
result: 40000 step, mAP 0.14
conclusion: 0.4 better than 1.0
