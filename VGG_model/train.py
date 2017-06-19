from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf

from vgg19_trainable import Vgg19
from vgg19_trainable import Train_Flags



train_flags = Train_Flags()


def _model_loss(vgg_class, refs_batch, poss_batch, negs_batch, test_batch):
    # Compute the moving average of all losses

    with tf.variable_scope(tf.get_variable_scope()):
        input_batch = tf.concat([refs_batch, poss_batch, negs_batch], 0)
        vgg_class.build(input_batch, test_batch, vgg_class.train_test_mode)

        vgg_class.calc_loss(vgg_class.fc7, train_flags.distance_alfa)

    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    # Compute the moving average of total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply([total_loss])

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)

    tf.summary.scalar('loss', total_loss)

    return total_loss


def train():
    """Train on dataset for a number of steps."""
    with tf.Graph().as_default():

        #build a VGG19 class object
        train_mode = tf.placeholder(tf.bool)

        vgg = Vgg19(vgg19_npy_path='./vgg19.npy', train_test_mode=train_mode)

        refs_batch, poss_batch, negs_batch = vgg.train_batch_inputs(train_flags.dataset_train_csv_file_path, train_flags.batch_size)
        test_batch = vgg.test_batch_inputs(train_flags.dataset_test_csv_file_path, train_flags.batch_size)

        loss = _model_loss(vgg, refs_batch, poss_batch, negs_batch, test_batch)


        global_step = tf.get_variable('global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        num_batches_per_epoch = (train_flags.num_per_epoch / train_flags.batch_size)

        decay_steps = int(num_batches_per_epoch * train_flags.num_epochs_per_decay)

        lr = tf.train.exponential_decay(train_flags.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        train_flags.learning_rate_decay_factor,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        vars_to_optimize = [v for v in tf.trainable_variables() if (v.name.startswith('fc') | v.name.startswith('conv4')) ]
        print '\nvariables to optimize'


        for v in vars_to_optimize:
            print v.name, v.get_shape().as_list()
            tf.summary.histogram(v.name, v)

        opt = tf.train.AdamOptimizer(lr)

        #grads = opt.compute_gradients(loss, var_list=vars_to_optimize)
        grads = opt.compute_gradients(loss, var_list=vars_to_optimize)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        train_op = tf.group(apply_gradient_op)

        summary_op = tf.summary.merge_all()

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter(train_flags.output_summary_path, graph=sess.graph)
        saver = tf.train.Saver(tf.global_variables())

        print('start training')
        for step in range(train_flags.max_step):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss], feed_dict={train_mode: True})

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                examples_per_sec = train_flags.batch_size / float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, duration))

            if step % 101 == 0:
                summary_str = sess.run(summary_op, feed_dict={train_mode: True})
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 5000 == 0 or (step + 1) == train_flags.max_step:
                checkpoint_path = os.path.join(train_flags.output_check_point_path, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

                test_dataset_features = np.zeros((train_flags.test_num, train_flags.test_output_feature_dim), dtype=float)
                batch_index = 0
                batch_size_test = train_flags.batch_size
                # do test: send every image in test.csv and get feature vector
                # save all feature vector in npy
                while True:
                    if(batch_index*batch_size_test < train_flags.test_num):
                        test_batch_output_feature = sess.run(vgg.fc7, feed_dict={train_mode: False})
                        test_dataset_features[(batch_index * batch_size_test) : ((batch_index + 1) * batch_size_test), 0:train_flags.test_output_feature_dim] = test_batch_output_feature

                        batch_index = batch_index + 1
                        print 'batch_index:', batch_index
                    else:
                        break
                test_dataset_features_filename = os.path.join(train_flags.output_test_features_path, 'test_features_step-%d.npy' % step)
                np.save(test_dataset_features_filename, test_dataset_features)


        coord.request_stop()
        coord.join(threads)
        sess.close()


def test():

    """Train on dataset for a number of steps."""
    with tf.Graph().as_default():

        #build a VGG19 class object
        train_mode = tf.placeholder(tf.bool)

        vgg = Vgg19(vgg19_npy_path='./vgg19.npy', train_test_mode=train_mode)

        refs_batch, poss_batch, negs_batch = vgg.train_batch_inputs(train_flags.dataset_train_csv_file_path, train_flags.batch_size)
        test_batch = vgg.test_batch_inputs(train_flags.dataset_test_csv_file_path, train_flags.batch_size)

        loss = _model_loss(vgg, refs_batch, poss_batch, negs_batch, test_batch)

        sess = tf.Session()

        init = tf.global_variables_initializer()

        sess.run(init)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        # Restore variables from disk.
        saver.restore(sess, train_flags.output_check_point_path + "/model.ckpt")
        print("Model restored.")

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        print('start testing')

        test_dataset_features = np.zeros((train_flags.test_num, train_flags.test_output_feature_dim), dtype=float)
        batch_index = 0
        # do test: send every image in test.csv and get feature vector
        # save all feature vector in npy
        while True:
            try:
                test_batch_output_feature = sess.run([vgg.fc7], feed_dict={train_mode: False})
                test_dataset_features[(batch_index * train_flags.batch_size) : \
                    (batch_index + 1) * train_flags.batch_size, : ] = test_batch_output_feature

                batch_index = batch_index + 1
                print 'process batch_index: ', batch_index

            except tf.errors.OutOfRangeError:
                break

        test_dataset_features_filename = os.path.join(train_flags.output_test_features_path, 'test_features_step-testing.npy')
        np.save(test_dataset_features_filename, test_dataset_features)

        coord.request_stop()
        coord.join(threads)
        sess.close()



train()
#test()
