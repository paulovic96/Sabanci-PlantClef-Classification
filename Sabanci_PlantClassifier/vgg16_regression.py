from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import os
#import urllib.request as urllib2
import math
slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

from datasets import dataset_utils
from datasets import imagenet
from nets import inception
from nets import vgg
from preprocessing import inception_preprocessing
from preprocessing import vgg_preprocessing
from datasets import plantclef2015
from datasets import plantclef2015_all_labels
from datasets import flowers
from datasets import plantVSnoplant
#import matplotlib.pyplot as plt
from preprocessing import sabanci_preprocessing

# Data
plant_data_dir = 'plantclef2015_all_labels'#'plantclef2015'

#run1
weight_decay = 0.0002
start_learning_rate = 0.001
updating_iteration_for_learning_rate =12000
updating_gamma =0.96
momentum = 0.9
num_classes = 1000
batch_size = 20


# Training Data
train_inceptionV1_dir = 'inception_finetuned'
train_vgg16_dir = 'vgg16_finetuned'
train_inceptionV1_bin_dir ='inception_bin_finetuned'

# Evaluation Data
eval_inceptionV1_dir = 'inception_evaluation'
eval_vgg16_dir = 'vgg16_evaluation'
eval_inceptionV1_bin_dir = 'inception_bin_evaluation'




train_vgg16_regression_eval_dir = 'vgg16_regression_eval_%s'
train_vgg16_regression_dir = 'vgg16_regression_%s'



vgg16_endpoints = ['vgg_16/conv1/conv1_1', 'vgg_16/conv1/conv1_2', 'vgg_16/pool1', 'vgg_16/conv2/conv2_1', 'vgg_16/conv2/conv2_2', 'vgg_16/pool2', 'vgg_16/conv3/conv3_1', 'vgg_16/conv3/conv3_2', 'vgg_16/conv3/conv3_3', 'vgg_16/pool3', 'vgg_16/conv4/conv4_1', 'vgg_16/conv4/conv4_2', 'vgg_16/conv4/conv4_3', 'vgg_16/pool4', 'vgg_16/conv5/conv5_1', 'vgg_16/conv5/conv5_2', 'vgg_16/conv5/conv5_3', 'vgg_16/pool5', 'vgg_16/fc6', 'vgg_16/fc7', 'vgg_16/fc8']


def vgg_16_base(inputs,
                 num_classes=1000,
                 is_training=True,
                 fc_conv_padding='VALID',
                 dropout_keep_prob=0.5,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='vgg_16',
                 final_endpoint = None):

    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            with tf.variable_scope('conv1'):
                endpoint = 'conv1_1'
                #net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.conv2d(inputs, 64, [3, 3], scope=endpoint)
                if final_endpoint == endpoint: return net
                endpoint = 'conv1_2'
                net = slim.conv2d(net, 64, [3, 3], scope=endpoint)
                if final_endpoint == endpoint: return net

            endpoint = 'pool1'
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            if final_endpoint == endpoint: return net

            with tf.variable_scope('conv2'):
                endpoint = 'conv2_1'
                net = slim.conv2d(net, 128, [3, 3], scope=endpoint)
                if final_endpoint == endpoint: return net
                endpoint = 'conv2_2'
                net = slim.conv2d(net, 128, [3, 3], scope=endpoint)
                if final_endpoint == endpoint: return net
            #net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')

            endpoint = 'pool2'
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            if final_endpoint == endpoint: return net

            with tf.variable_scope('conv3'):
                endpoint = 'conv3_1'
                net = slim.conv2d(net, 256, [3, 3], scope=endpoint)
                if final_endpoint == endpoint: return net
                endpoint = 'conv3_2'
                net = slim.conv2d(net, 256, [3, 3], scope=endpoint)
                if final_endpoint == endpoint: return net
                endpoint = 'conv3_3'
                net = slim.conv2d(net, 256, [3, 3], scope=endpoint)
                if final_endpoint == endpoint: return net
            #net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')


            endpoint = 'pool3'
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            if final_endpoint == endpoint: return net

            with tf.variable_scope('conv4'):
                endpoint = 'conv4_1'
                net = slim.conv2d(net, 512, [3, 3], scope=endpoint)
                if final_endpoint == endpoint: return net
                endpoint = 'conv4_2'
                net = slim.conv2d(net, 512, [3, 3], scope=endpoint)
                if final_endpoint == endpoint: return net
                endpoint = 'conv4_3'
                net = slim.conv2d(net, 512, [3, 3], scope=endpoint)
                if final_endpoint == endpoint: return net
            #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')

            endpoint = 'pool4'
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            if final_endpoint == endpoint: return net

            with tf.variable_scope('conv5'):
                endpoint = 'conv5_1'
                net = slim.conv2d(net, 512, [3, 3], scope=endpoint)
                if final_endpoint == endpoint: return net
                endpoint = 'conv5_2'
                net = slim.conv2d(net, 512, [3, 3], scope=endpoint)
                if final_endpoint == endpoint: return net
                endpoint = 'conv5_3'
                net = slim.conv2d(net, 512, [3, 3], scope=endpoint)
                if final_endpoint == endpoint: return net
            #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')


            endpoint = 'pool5'
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            if final_endpoint == endpoint: return net

            # Use conv2d instead of fully_connected layers.
            endpoint = 'fc6'
            net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
            if final_endpoint == endpoint: return net
            endpoint = 'dropout6'
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                scope='dropout6')
            if final_endpoint == endpoint: return net
            endpoint = 'fc7'
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            if final_endpoint == endpoint: return net
            endpoint = 'dropout7'
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                scope='dropout7')
            if final_endpoint == endpoint: return net
            endpoint = 'fc8'
            net = slim.conv2d(net, num_classes, [1, 1],
                                activation_fn=None,
                                normalizer_fn=None,
                                scope='fc8')
            if final_endpoint == endpoint: return net
            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                endpoint = 'fc8/squeezed'
                net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                if final_endpoint == endpoint: return net
                end_points[sc.name + '/fc8'] = net

            return net, end_points






def vgg_16(inputs,
                 num_classes=1000,
                 is_training=True,
                 fc_conv_padding='VALID',
                 dropout_keep_prob=0.5,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='vgg_16',
                 regression_point = None):

    net, _ = vgg_16_base(inputs,num_classes=num_classes,scope=scope, final_endpoint=regression_point,is_training=is_training, dropout_keep_prob = dropout_keep_prob, spatial_squeeze = spatial_squeeze)  #genus 516 #family 124
    with tf.variable_scope('Regression'):
        if is_training:
            net = slim.dropout(net,dropout_keep_prob, scope='Dropout_0b')
        regression = slim.flatten(net)
        regression_layer = "%s_fc" % regression_point
        regression = slim.fully_connected(regression, num_classes, activation_fn=None, normalizer_fn=None, scope=regression_layer)

        return regression


def get_init_fn_vgg(layer):

    layer_dir = os.path.join(train_vgg16_regression_dir, layer)
    if tf.train.latest_checkpoint(layer_dir):
        tf.logging.info(
            'checkpoint exists in %s'
            % layer_dir)
        tf.logging.info(
            'Fine-tuning from %s' % layer_dir)
        return None

    else:
        tf.logging.info('load predtrained model from %s' % train_vgg16_dir)
        checkpoint_exclude_scopes = ['Regression/%s_fc' % layer]
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)

        return slim.assign_from_checkpoint_fn(tf.train.latest_checkpoint(train_vgg16_dir), variables_to_restore)








def loss(logits, labels, num_classes, head=None):
    """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.up as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
      head: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes
    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):
        epsilon = tf.constant(value=1e-4)

        softmax = tf.nn.softmax(logits) + epsilon

        if head is not None:
            cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax),
                                           head), reduction_indices=[1])
        else:
            cross_entropy = -tf.reduce_sum(
                labels * tf.log(softmax), reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='xentropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss







def train_vgg16_regression(log_steps,save_summaries_sec,save_interval_secs,num_iterations = 8863, layer=None, split=None, batch_size=8, classes = "species"):
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        global_step = slim.get_or_create_global_step()

        dataset = plantclef2015_all_labels.get_split(split, plant_data_dir)
        #images,labels = load_batch(dataset, batch_size = batch_size, k=num_patches_inception, r=r_rotations_inception)


        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32,common_queue_min=8, shuffle = False)

        image_raw, label = data_provider.get(['image', "label_%s" % classes])

        image = tf.image.per_image_standardization(image_raw)
        image = tf.image.resize_images(image, [224, 224])

        images, labels = tf.train.batch([image, label],batch_size=batch_size, shapes = [tf.TensorShape([tf.Dimension(224), tf.Dimension(224), tf.Dimension(3)]), tf.TensorShape([])],num_threads=1, capacity=2 * 20)


        # Create the models

        if classes == "genus":
            with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
             regression = vgg_16(images, num_classes=516, is_training=True, regression_point = layer)

            one_hot_genus_labels = slim.one_hot_encoding(labels, 516)
            loss(regression, one_hot_genus_labels, 516)

        elif classes == "family":
            with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
             regression = vgg_16(images, num_classes=124, is_training=True, regression_point = layer)

            one_hot_family_labels = slim.one_hot_encoding(labels, 124)
            loss(regression, one_hot_family_labels, 124)

        elif classes == "organ":
            with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
             regression = vgg_16(images, num_classes=7, is_training=True, regression_point = layer)

            one_hot_organ_labels = slim.one_hot_encoding(labels, 7)
            loss(regression, one_hot_organ_labels, 7)

        else:
            with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
             regression = vgg_16(images, num_classes=1000, is_training=True, regression_point = layer)

            one_hot_species_labels = slim.one_hot_encoding(labels, 1000)
            loss(regression, one_hot_species_labels, 1000)


        #tf.losses.mean_squared_error(one_hot_labels,regression,scope= "%s_fc" % layer)
        total_loss = slim.losses.get_total_loss()


        summaries.add(tf.summary.scalar('losses/%s_fc' % layer , total_loss))
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,updating_iteration_for_learning_rate, updating_gamma, staircase=True)

        optimizer = tf.train.MomentumOptimizer(learning_rate= learning_rate, momentum=momentum)

        train_op = slim.learning.create_train_op(total_loss, optimizer,variables_to_train=slim.get_variables(scope='Regression/%s_fc' % layer))#'InceptionV1/%s' % fc))
            #train_ops.append(train_op)


        #train_op = tf.stack(train_ops,0)
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        train_dir = train_vgg16_regression_dir % classes

        log_dir = os.path.join(train_dir, layer)



        ## Run the training:
        slim.learning.train(
            train_op,
            logdir=log_dir,
            log_every_n_steps=log_steps,
            global_step=global_step,
            number_of_steps= num_iterations,
            summary_op=summary_op,
            init_fn=get_init_fn_vgg(layer),
            save_summaries_secs=save_summaries_sec,
            save_interval_secs=save_interval_secs)

    print('Finished training.')



def evaluate_vgg16_regression(batch_size,layer,classes="species"):
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        global_step = slim.get_or_create_global_step()

        # Summary


        dataset = plantclef2015_all_labels.get_split('validation', plant_data_dir)
        #dataset = plantclef2015.get_split('train', plant_data_dir)
        #images,labels = load_batch(dataset, batch_size = batch_size, k=num_patches_inception, r=r_rotations_inception)


        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32,common_queue_min=8, shuffle = True)

        image_raw, label = data_provider.get(['image', 'label_%s' % classes])

        image = tf.image.per_image_standardization(image_raw)
        image = tf.image.resize_images(image, [224, 224])

        images, labels = tf.train.batch([image, label],batch_size=1, shapes = [tf.TensorShape([tf.Dimension(224), tf.Dimension(224), tf.Dimension(3)]), tf.TensorShape([])],num_threads=1, capacity=2 * 20)


        # Create the models
        if classes == "genus":
            with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
             regression = vgg_16(images, num_classes=516, is_training=False, regression_point = layer)


        elif classes == "family":
            with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
             regression = vgg_16(images, num_classes=124, is_training=False, regression_point = layer)


        elif classes == "organ":
            with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
             regression = vgg_16(images, num_classes=7, is_training=False, regression_point = layer)


        else:
            with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
             regression = vgg_16(images, num_classes=1000, is_training=False, regression_point = layer)


        train_dir = train_vgg16_regression_dir % classes
        train_dir = os.path.join(train_dir, layer)


        total_output = []
        total_labels = []


        train_dir = os.path.join(train_vgg16_regression_dir, layer)
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(train_dir))
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(batch_size):
                print('step: %d/%d' % (i, batch_size))
                o, l= sess.run([regression, labels])


                total_output.append(o[0])
                total_labels.append(l[0])


            coord.request_stop()
            coord.join(threads)


            total_output = tf.stack(total_output,0)
            total_output = tf.nn.softmax(total_output)

            total_labels = tf.constant(total_labels)
            label, pred =  sess.run([total_labels,tf.argmax(total_output, axis=1)])


            print(layer)
            top1_op = tf.nn.in_top_k(total_output, total_labels, 1)
            top1_acc = sess.run(tf.reduce_mean(tf.cast(top1_op, tf.float32)))
            print(top1_acc)


            top5_op = tf.nn.in_top_k(total_output, total_labels, 5)
            top5_acc = sess.run(tf.reduce_mean(tf.cast(top5_op, tf.float32)))
            print(top5_acc)

            accuracy1_sum = tf.summary.scalar('top1_accuracy', top1_acc)
            accuracy5_sum = tf.summary.scalar('top5_accuracy', top5_acc)

            eval_dir = train_vgg16_regression_eval_dir % classes
            log_dir = os.path.join(eval_dir, layer)

            accuracy1, accuracy5,  step = sess.run([accuracy1_sum,accuracy5_sum, global_step])
            writer = tf.summary.FileWriter(log_dir)
            writer.add_summary(accuracy1, step)
            writer.add_summary(accuracy5, step)

            return top1_acc,top5_acc



