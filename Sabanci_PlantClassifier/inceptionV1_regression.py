from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import os
import urllib.request as urllib2
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
import matplotlib.pyplot as plt
from preprocessing import sabanci_preprocessing

# Data
plant_data_dir = 'plantclef2015_all_labels'
plant_test_data_dir = "plantclef2015_test_all_labels"

#run1
weight_decay = 0.0002
start_learning_rate = 0.001
updating_iteration_for_learning_rate =12000
updating_gamma =0.96
momentum = 0.9
#num_classes = 1000
batch_size = 20


# Training Data
train_inceptionV1_dir = 'inception_finetuned'
train_vgg16_dir = 'vgg16_finetuned'
train_inceptionV1_bin_dir ='inception_bin_finetuned'

# Evaluation Data
eval_inceptionV1_dir = 'inception_evaluation'
eval_vgg16_dir = 'vgg16_evaluation'
eval_inceptionV1_bin_dir = 'inception_bin_evaluation'



train_inceptionV1_regression_dir = 'inceptionV1_regression_%s'
train_inceptionV1_regression_eval_dir = 'inceptionV1_regression_eval_%s'


def inception_v1_base(inputs,
                      final_endpoint=None,
                      scope='InceptionV1'):
  """Defines the Inception V1 base architecture.

  This architecture is defined in:
    Going deeper with convolutions
    Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    http://arxiv.org/pdf/1409.4842v1.pdf.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
      'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
      'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']
    scope: Optional variable_scope.

  Returns:
    A dictionary from components of the network to the corresponding activation.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values.
  """
  end_points = {}
  with tf.variable_scope(scope, 'InceptionV1', [inputs]):
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_initializer=trunc_normal(0.01)):
      with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                          stride=1, padding='SAME'):
        end_point = 'Conv2d_1a_7x7'
        net = slim.conv2d(inputs, 64, [7, 7], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint==  end_point: return net, end_points
        end_point = 'MaxPool_2a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'Conv2d_2b_1x1'
        net = slim.conv2d(net, 64, [1, 1], scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'Conv2d_2c_3x3'
        net = slim.conv2d(net, 192, [3, 3], scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'MaxPool_3a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_3b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_3c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 192, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool_4a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 208, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4d'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4e'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 144, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 288, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4f'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool_5a_2x2'
        net = slim.max_pool2d(net, [2, 2], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0a_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 384, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_v1(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV1',
                 regression_point = None):
  """Defines the Inception V1 architecture.

  This architecture is defined in:

    Going deeper with convolutions
    Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    http://arxiv.org/pdf/1409.4842v1.pdf.

  The default image size used to train this network is 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    regression_point = point after which a FC-Layer is pluged in

  Returns:
    regression: the pre-softmax activations after intermediate layer, a tensor of size
      [batch_size, num_classes]
  """
  # Final pooling and prediction
  with tf.variable_scope(scope, 'InceptionV1', [inputs, num_classes],
                         reuse=reuse) as scope:
    if regression_point == "Mixed_5c": # transferlearning
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, _ = inception_v1_base(inputs,final_endpoint= regression_point, scope=scope)
            with tf.variable_scope('Regression'):
                net = slim.avg_pool2d(net, [7, 7], stride=1, scope='AvgPool_0a_7x7')
                net = slim.dropout(net,
                                dropout_keep_prob, scope='Dropout_0b')
                regression_layer = "%s_fc" % regression_point
                regression = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                    normalizer_fn=None, scope=regression_layer)
                if spatial_squeeze:
                    regression = tf.squeeze(regression, [1, 2], name='SpatialSqueeze')

    else: # plug in FC-Layer after
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, _ = inception_v1_base(inputs,final_endpoint= regression_point,scope=scope)  #genus 516 #family 124
            with tf.variable_scope('Regression'):

                if is_training:
                    net = slim.dropout(net,dropout_keep_prob, scope='Dropout_0b')
                regression = slim.flatten(net)
                regression_layer = "%s_fc" % regression_point
                regression = slim.fully_connected(regression, num_classes, activation_fn=None, normalizer_fn=None, scope=regression_layer)
  return regression




def get_init_fn_V1(layer,classes):
    layer_dir = os.path.join(train_inceptionV1_regression_dir % classes, layer)
    if tf.train.latest_checkpoint(layer_dir):
        tf.logging.info(
            'checkpoint exists in %s'
            % layer_dir)
        tf.logging.info(
            'Fine-tuning from %s' % layer_dir)
        return None

    else:
        tf.logging.info('load predtrained model from %s' % train_inceptionV1_dir)
        checkpoint_exclude_scopes = ['InceptionV1/Regression/%s_fc' % layer]
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

        return slim.assign_from_checkpoint_fn(tf.train.latest_checkpoint(train_inceptionV1_dir), variables_to_restore)








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





def train_inceptionV1_regression(log_steps,save_summaries_sec,save_interval_secs,num_iterations = 8863, layer=None, split=None, batch_size=8, classes = "species"):
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        global_step = slim.get_or_create_global_step()

        dataset = plantclef2015_all_labels.get_split(split, plant_data_dir)
        #images,labels = load_batch(dataset, batch_size = batch_size, k=num_patches_inception, r=r_rotations_inception)


        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32,common_queue_min=8, shuffle = True) #False

        image_raw, label = data_provider.get(['image', "label_%s" % classes])

        image = tf.image.per_image_standardization(image_raw)
        image = tf.image.resize_images(image, [224, 224])

        images, labels = tf.train.batch([image, label],batch_size=batch_size, shapes = [tf.TensorShape([tf.Dimension(224), tf.Dimension(224), tf.Dimension(3)]), tf.TensorShape([])],num_threads=1, capacity=2 * 20)


        # Create the model
        if classes == "genus":
            with slim.arg_scope(inception.inception_v1_arg_scope(weight_decay=weight_decay)):
                regression = inception_v1(images, num_classes=516, is_training=True, regression_point=layer)

            one_hot_genus_labels = slim.one_hot_encoding(labels, 516)
            loss(regression, one_hot_genus_labels, 516)

        elif classes == "family":

            with slim.arg_scope(inception.inception_v1_arg_scope(weight_decay=weight_decay)):
                regression = inception_v1(images, num_classes=124, is_training=True, regression_point=layer)


            one_hot_family_labels = slim.one_hot_encoding(labels, 124)
            loss(regression, one_hot_family_labels, 124)

        elif classes == "organ":

            with slim.arg_scope(inception.inception_v1_arg_scope(weight_decay=weight_decay)):
                regression = inception_v1(images, num_classes=7, is_training=True, regression_point=layer)


            one_hot_organ_labels = slim.one_hot_encoding(labels, 7)
            loss(regression, one_hot_organ_labels, 7)

        else:
            with slim.arg_scope(inception.inception_v1_arg_scope(weight_decay=weight_decay)):
                regression = inception_v1(images, num_classes=1000, is_training=True, regression_point=layer)


            one_hot_species_labels = slim.one_hot_encoding(labels, 1000)
            loss(regression, one_hot_species_labels, 1000)




        #slim.losses.softmax_cross_entropy(one_hot_labels,regression)
        #tf.losses.mean_squared_error(one_hot_labels,regression,scope= "%s_fc" % layer)
        total_loss = slim.losses.get_total_loss()


        summaries.add(tf.summary.scalar('losses/%s_fc' % layer , total_loss))
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,updating_iteration_for_learning_rate, updating_gamma, staircase=True)

        optimizer = tf.train.MomentumOptimizer(learning_rate= learning_rate, momentum=momentum)

        train_op = slim.learning.create_train_op(total_loss, optimizer,variables_to_train=slim.get_variables(scope='InceptionV1/Regression/%s_fc' % layer))#'InceptionV1/%s' % fc))



        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        train_dir = train_inceptionV1_regression_dir % classes

        log_dir = os.path.join(train_dir, layer)
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True

        ## Run the training:
        slim.learning.train(
            train_op,
            logdir=log_dir,
            log_every_n_steps=log_steps,
            global_step=global_step,
            number_of_steps= num_iterations,
            summary_op=summary_op,
            init_fn=get_init_fn_V1(layer,classes),
            save_summaries_secs=save_summaries_sec,
            save_interval_secs=save_interval_secs,
            session_config = sess_config)

    print('Finished training.')



def evaluate_inceptionV1_regression(batch_size,layer, classes="species"):
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        global_step = slim.get_or_create_global_step()


        dataset = plantclef2015_all_labels.get_split('validation', plant_data_dir)


        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32,common_queue_min=8, shuffle = True)

        image_raw, label = data_provider.get(['image', 'label_%s' % classes])

        image = tf.image.per_image_standardization(image_raw)
        image = tf.image.resize_images(image, [224, 224])

        images, labels = tf.train.batch([image, label],batch_size=1, shapes = [tf.TensorShape([tf.Dimension(224), tf.Dimension(224), tf.Dimension(3)]), tf.TensorShape([])],num_threads=1, capacity=2 * 20)


        # Create the models
        if classes == "genus":
            with slim.arg_scope(inception.inception_v1_arg_scope(weight_decay=weight_decay)):
                regression = inception_v1(images, num_classes=516, is_training=False, regression_point=layer)

        elif classes == "family":
            with slim.arg_scope(inception.inception_v1_arg_scope(weight_decay=weight_decay)):
                regression = inception_v1(images, num_classes=124, is_training=False, regression_point=layer)

        elif classes == "organ":
            with slim.arg_scope(inception.inception_v1_arg_scope(weight_decay=weight_decay)):
                regression = inception_v1(images, num_classes=7, is_training=False, regression_point=layer)

        else:
            with slim.arg_scope(inception.inception_v1_arg_scope(weight_decay=weight_decay)):
                regression = inception_v1(images, num_classes=1000, is_training=False, regression_point=layer)


        total_output = []
        total_labels = []

        train_dir = train_inceptionV1_regression_dir % classes
        train_dir = os.path.join(train_dir, layer)


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

            eval_dir = train_inceptionV1_regression_eval_dir % classes
            log_dir = os.path.join(eval_dir, layer)

            accuracy1, accuracy5,  step = sess.run([accuracy1_sum,accuracy5_sum, global_step])
            writer = tf.summary.FileWriter(log_dir)
            writer.add_summary(accuracy1, step)
            writer.add_summary(accuracy5, step)

            return top1_acc,top5_acc






def regression_results(classes):
  """ Write down the accuracy for a class/category
  Args:
    classes: one of "species", "genus", "family", "organ"
  """
    layer_filename = "regression_%s.txt" % classes
    with tf.gfile.Open(layer_filename, 'w') as f:
        f.write('Layer    Class    Top1    Top5\n')
        for endpoint in endpoints:
            class_top1, class_top5 = evaluate_inceptionV1_regression(20000,endpoint,classes=classes)
            f.write("%s    %s    %f    %f\n" % (endpoint,classes,class_top1,class_top5))
