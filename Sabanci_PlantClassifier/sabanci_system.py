from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import os
import urllib.request as urllib2
import math
slim = tf.contrib.slim


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
plantVSnoplant_data_dir ='plantVSnoplant'


# Pre-trained Data
inceptionV3_url = "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
inceptionV1_url = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"
vgg16_url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
checkpoints_inceptionV3_dir = 'checkpoints_inceptionV3'
checkpoints_inceptionV1_dir = 'checkpoints_V1'
checkpoints_vgg16_dir ='checkpoints_vgg16'

# Download pretrained data
# dataset_utils.download_and_uncompress_tarball(inceptionV3_url, checkpoints_inceptionV3_dir)
# dataset_utils.download_and_uncompress_tarball(inceptionV1_url, checkpoints_inceptionV1_dir)
# dataset_utils.download_and_uncompress_tarball(vgg16_url, checkpoints_vgg16_dir)




# Training Data
train_inceptionV1_dir = 'inception_finetuned'
train_vgg16_dir = 'vgg16_finetuned'
train_inceptionV1_bin_dir ='inception_bin_finetuned'

# Evaluation Data
eval_inceptionV1_dir = 'inception_evaluation'
eval_vgg16_dir = 'vgg16_evaluation'
eval_inceptionV1_bin_dir = 'inception_bin_evaluation'


#if not tf.gfile.Exists(train_inceptionV1_dir):
    #tf.gfile.MakeDirs(train_inceptionV1_dir)

#if not tf.gfile.Exists(train_vgg16_dir):
    #tf.gfile.MakeDirs(train_vgg16_dir)

#if not tf.gfile.Exists(train_vgg16_dir):
    #tf.gfile.MakeDirs(train_vgg16_dir)

#if not tf.gfile.Exists(eval_inceptionV1_dir):
    #tf.gfile.MakeDirs(eval_inceptionV1_dir)

#if not tf.gfile.Exists(eval_vgg16_dir):
    #tf.gfile.MakeDirs(eval_vgg16_dir)

#if not tf.gfile.Exists(eval_inceptionV1_bin_dir):
    #tf.gfile.MakeDirs(eval_inceptionV1_bin_dir)




#--------------------------------------------------DATA--------------------------------------------------#
#--------------------------------------------------------------------------------------------------------#
def load_batch(dataset, k, r, batch_size=20, height=224, width=224, is_training=True, is_testing=False):
    """Loads a single batch of data.

    Args:
      dataset: The dataset to load.
      k: The number of square patches from orginal image
      r: The number of degree the orginal image is rotated
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.

    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """




    if is_training:

        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32,common_queue_min=8, shuffle = True)

        image_raw, label = data_provider.get(['image', 'label'])
        #image_raw, label = data_provider.get(['image', "label_%s" % 'species'])
        image = sabanci_preprocessing.get_patches(image_raw, k, 0.6, 0.8, r, False)

        images, labels = tf.train.batch(
          [image, label],
          batch_size=batch_size,
          shapes = [tf.TensorShape([tf.Dimension(224), tf.Dimension(224), tf.Dimension(3)]), tf.TensorShape([])],
          num_threads=1,
          capacity=2 * batch_size)



    else:
        if is_testing:
            data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32,common_queue_min=8, shuffle = False)

            image_raw, label = data_provider.get(['image', 'image_id'])

            pre_images = sabanci_preprocessing.get_patches(image_raw, k, 0.6, 0.8, r, True)
            pre_labels = [label for i in range(5*(k + 2 * len(r) + 1))]

            images = tf.stack(pre_images,0)
            labels = tf.stack(pre_labels,0)

        else:
            data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32,common_queue_min=8, shuffle = False)

            image_raw, label = data_provider.get(['image', 'label'])
            #image_raw, label = data_provider.get(['image', "label_%s" % 'species'])

            pre_images = sabanci_preprocessing.get_patches(image_raw, k, 0.6, 0.8, r, True)
            pre_labels = [label for i in range(5*(k + 2 * len(r) + 1))]

            images = tf.stack(pre_images,0)
            labels = tf.stack(pre_labels,0)

    return images, labels




#-----------------------------------------------------------Initialization----------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#
def get_init_fn_V1():
    if tf.train.latest_checkpoint(train_inceptionV1_dir):
        tf.logging.info(
            'checkpoint exists in %s'
            % train_inceptionV1_dir)
        tf.logging.info(
            'Fine-tuning from %s' % train_inceptionV1_dir)
        return None

    else:
        tf.logging.info('warm-starting with downloaded checkpoints in %s' % checkpoints_inceptionV1_dir)
        checkpoint_exclude_scopes = ["InceptionV1/Logits", "InceptionV1/AuxLogits"]
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

        return slim.assign_from_checkpoint_fn(os.path.join(checkpoints_inceptionV1_dir, 'inception_v1.ckpt'), variables_to_restore)



def get_init_fn_vgg():
    if tf.train.latest_checkpoint(train_vgg16_dir):
        tf.logging.info(
            'checkpoint exists in %s'
            % train_vgg16_dir)
        tf.logging.info(
            'Fine-tuning from %s' % train_vgg16_dir)
        return None

    else:
        tf.logging.info('warm-starting with downloaded checkpoints in %s' % checkpoints_vgg16_dir)
        checkpoint_exclude_scopes = ["vgg_16/fc8","vgg_16/fc8/squeezed"]
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

        return slim.assign_from_checkpoint_fn(os.path.join(checkpoints_vgg16_dir, 'vgg_16.ckpt'), variables_to_restore)



def get_init_fn_V1_bin():

    if tf.train.latest_checkpoint(train_inceptionV1_bin_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % train_inceptionV1_bin_dir)
        tf.logging.info(
            'Fine-tuning from %s' % train_inceptionV1_bin_dir)
        return None

    else:
        tf.logging.info('warm-starting with pretrained checkpoints in %s' % train_inceptionV1_dir)
        checkpoint_exclude_scopes = ["InceptionV1/Logits", "InceptionV1/AuxLogits"]
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



#----------------------------------------Training----------------------------------------------------#
#----------------------------------------------------------------------------------------------------#

#################
##Training Info##
#################

#run1
weight_decay = 0.0002
start_learning_rate = 0.001
updating_iteration_for_learning_rate =12000
updating_gamma =0.96
momentum = 0.9
num_classes = 1000
batch_size = 20

# 1.1
# Training set: plantclef2015 training set with 70904 images
num_patches_inception = 9
r_rotations_inception = [10]#[10,20]
num_iterations_inception = 800000 #600000

num_patches_vgg = 5
r_rotations_vgg = [10]
num_iterations_vgg = 500000

num_iterations_V1_bin = 100000

# 1.2
# Training set Inception: plantclef 2015 training+test+validation/2 merged
#num_patches_inception = 5
#r_rotations_inception = [10]
#num_iterations_inception = 200000



def loss(logits, labels, head=None):
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




def train_vgg16(log_steps,save_summaries_sec,save_interval_secs,num_iterations = num_iterations_vgg):
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        global_step = slim.get_or_create_global_step()

        #dataset = plantclef2015.get_split('train', plant_data_dir)
        dataset = plantclef2015_all_labels.get_split('train', plant_data_dir)

        images,labels = load_batch(dataset, batch_size = batch_size, k=num_patches_vgg, r=r_rotations_vgg)

        # Add Images to summaries
        summaries.add(tf.summary.image("input_images", images, batch_size))

        # Create the models
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
             logits, _ = vgg.vgg_16(images, num_classes=1000, is_training=False)


        # Specify the loss function:
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

        loss(logits, one_hot_labels)
        #slim.losses.softmax_cross_entropy(logits, one_hot_labels)
        #tf.losses.softmax_cross_entropy(one_hot_labels, logits)
        total_loss = slim.losses.get_total_loss()

        # Create some summaries to visualize the training process:
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))
        summaries.add(tf.summary.scalar('losses/Total_Loss', total_loss))

        # Specify the optimizer and create the train op:
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,updating_iteration_for_learning_rate, updating_gamma, staircase=True)

        optimizer = tf.train.MomentumOptimizer(learning_rate= learning_rate, momentum=momentum)

        train_op = slim.learning.create_train_op(total_loss, optimizer)

        summaries.add(tf.summary.scalar('training/Learning_Rate', learning_rate))

        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        # Run the training:
        final_loss = slim.learning.train(
            train_op,
            logdir=train_vgg16_dir,
            log_every_n_steps=log_steps,
            global_step=global_step,
            number_of_steps= num_iterations,
            summary_op=summary_op,
            init_fn=get_init_fn_vgg(),
            save_summaries_secs=save_summaries_sec,
            save_interval_secs=save_interval_secs)

    print('Finished training. Last batch loss %f' % final_loss)


def train_inceptionV1(log_steps,save_summaries_sec,save_interval_secs,num_iterations = num_iterations_inception):
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        global_step = slim.get_or_create_global_step()

        dataset = plantclef2015.get_split('train', plant_data_dir)
        images,labels = load_batch(dataset, batch_size = batch_size, k=num_patches_inception, r=r_rotations_inception)

        # Add Images to summaries
        summaries.add(tf.summary.image("input_images", images, batch_size))

        # Create the models
        with slim.arg_scope(inception.inception_v1_arg_scope(weight_decay=weight_decay)):
            logits, _ = inception.inception_v1(images, num_classes=1000, is_training=True)


        # Specify the loss function:
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

        loss(logits, one_hot_labels)
        #slim.losses.softmax_cross_entropy(logits, one_hot_labels)
        total_loss = slim.losses.get_total_loss()

        # Create some summaries to visualize the training process:
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))
        summaries.add(tf.summary.scalar('losses/Total_Loss', total_loss))

        # Specify the optimizer and create the train op:
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,updating_iteration_for_learning_rate, updating_gamma, staircase=True)

        optimizer = tf.train.MomentumOptimizer(learning_rate= learning_rate, momentum=momentum)

        train_op = slim.learning.create_train_op(total_loss, optimizer)

        summaries.add(tf.summary.scalar('training/Learning_Rate', learning_rate))

        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        # Run the training:
        final_loss = slim.learning.train(
            train_op,
            logdir=train_inceptionV1_dir,
            log_every_n_steps=log_steps,
            global_step=global_step,
            number_of_steps= num_iterations,
            summary_op=summary_op,
            init_fn=get_init_fn_V1(),
            save_summaries_secs=save_summaries_sec,
            save_interval_secs=save_interval_secs)

    print('Finished training. Last batch loss %f' % final_loss)


def train_inceptionV1_bin(log_steps,save_summaries_sec,save_interval_secs,num_iterations = 100000):
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        global_step = slim.get_or_create_global_step()

        dataset = plantVSnoplant.get_split('train', plantVSnoplant_data_dir)
        images,labels = load_batch(dataset, batch_size = batch_size, k=0, r=[])

        summaries.add(tf.summary.image("input_images", images, 20))
        # Create the models
        with slim.arg_scope(inception.inception_v1_arg_scope(weight_decay=weight_decay)):
            logits, _ = inception.inception_v1(images, num_classes=2, is_training=True)

        # Specify the loss function:
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)


        loss(logits, one_hot_labels)
        #slim.losses.softmax_cross_entropy(logits, one_hot_labels)
        total_loss = slim.losses.get_total_loss()

        # Create some summaries to visualize the training process:

        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))
        summaries.add(tf.summary.scalar('losses/Total_Loss', total_loss))

        # Specify the optimizer and create the train op:
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,updating_iteration_for_learning_rate, updating_gamma, staircase=True)

        optimizer = tf.train.MomentumOptimizer(learning_rate= learning_rate, momentum=momentum)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        summaries.add(tf.summary.scalar('training/Learning_Rate', learning_rate))

        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        # Run the training:
        final_loss = slim.learning.train(
            train_op,
            logdir=train_inceptionV1_bin_dir,
            init_fn=get_init_fn_V1_bin(),
            log_every_n_steps=log_steps,
            global_step=global_step,
            number_of_steps= num_iterations,
            summary_op=summary_op,
            save_summaries_secs=save_summaries_sec,
            save_interval_secs=save_interval_secs)

    print('Finished training. Last batch loss %f' % final_loss)



#----------------------------------------Evaluation----------------------------------------------------#
#------------------------------------------------------------------------------------------------------#


def evaluate_vgg16(batch_size):
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        global_step = slim.get_or_create_global_step()


        dataset = plantclef2015.get_split('validation', plant_data_dir)
        images,labels = load_batch(dataset, batch_size = batch_size, k=num_patches_vgg, r=r_rotations_vgg, is_training =False)

        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
             logits, _ = vgg.vgg_16(images, num_classes=1000, is_training=False)


        total_output = []
        total_labels = []
        total_images = []

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            saver = tf. train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(train_vgg16_dir))
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in range(batch_size):
                print('step: %d/%d' % (i, batch_size))
                o, l , image= sess.run([logits, labels, images[0]])
                o = tf.reduce_sum(o, 0)/float(40)
                total_output.append(o)
                total_labels.append(l[0])
                total_images.append(image)
            coord.request_stop()
            coord.join(threads)


            total_output = tf.stack(total_output,0)
            total_output = tf.nn.softmax(total_output)
            labels = tf.constant(total_labels)
            total_images = sess.run(tf.stack(total_images,0))

            top1_op = tf.nn.in_top_k(total_output, labels, 1)
            top1_acc = sess.run(tf.reduce_mean(tf.cast(top1_op, tf.float32)))
            print(top1_acc)


            top5_op = tf.nn.in_top_k(total_output, labels, 5)
            top5_acc = sess.run(tf.reduce_mean(tf.cast(top5_op, tf.float32)))
            print(top5_acc)

            accuracy1_sum = tf.summary.scalar('top1_accuracy', top1_acc)
            accuracy5_sum = tf.summary.scalar('top5_accuracy', top5_acc)
            images_sum = tf.summary.image("input_images", total_images, batch_size)

            accuracy1, accuracy5, image_batch, step = sess.run([accuracy1_sum,accuracy5_sum,images_sum, global_step])
            writer = tf.summary.FileWriter(eval_vgg16_dir)
            writer.add_summary(accuracy1, step)
            writer.add_summary(accuracy5, step)
            writer.add_summary(image_batch)
            #return top1_acc, top5_acc



def evaluate_inceptionV1(batch_size):
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        global_step = slim.get_or_create_global_step()

        # Summary


        dataset = plantclef2015_all_labels.get_split('validation', plant_data_dir)
        images,labels = load_batch(dataset, batch_size = batch_size, k=num_patches_inception, r=r_rotations_inception, is_training =False)

        with slim.arg_scope(inception.inception_v1_arg_scope(weight_decay=weight_decay)):
                logits, _ = inception.inception_v1(images, num_classes=1000, is_training=False)


        total_output = []
        total_labels = []
        total_images = []

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            saver = tf. train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(train_inceptionV1_dir))
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in range(batch_size):
                print('step: %d/%d' % (i, batch_size))
                o, l , image= sess.run([logits, labels, images[0]])
                o = tf.reduce_sum(o, 0)/float(70)
                total_output.append(o)
                total_labels.append(l[0])
                total_images.append(image)
            coord.request_stop()
            coord.join(threads)


            total_output = tf.stack(total_output,0)
            total_output = tf.nn.softmax(total_output)
            labels = tf.constant(total_labels)
            total_images = sess.run(tf.stack(total_images,0))

            top1_op = tf.nn.in_top_k(total_output, labels, 1)
            top1_acc = sess.run(tf.reduce_mean(tf.cast(top1_op, tf.float32)))
            print(top1_acc)


            top5_op = tf.nn.in_top_k(total_output, labels, 5)
            top5_acc = sess.run(tf.reduce_mean(tf.cast(top5_op, tf.float32)))
            print(top5_acc)

            accuracy1_sum = tf.summary.scalar('top1_accuracy', top1_acc)
            accuracy5_sum = tf.summary.scalar('top5_accuracy', top5_acc)
            images_sum = tf.summary.image("input_images", total_images, batch_size)

            accuracy1, accuracy5, image_batch, step = sess.run([accuracy1_sum,accuracy5_sum,images_sum, global_step])
            writer = tf.summary.FileWriter(eval_inceptionV1_dir)
            writer.add_summary(accuracy1, step)
            writer.add_summary(accuracy5, step)
            writer.add_summary(image_batch)
            #return top1_acc, top5_acc


def evaluate_inceptionV1_bin(batch_size):
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        global_step = slim.get_or_create_global_step()

        # Summary


        dataset = plantVSnoplant.get_split('validation', plantVSnoplant_data_dir)
        images,labels = load_batch(dataset, batch_size = batch_size,  k=0, r=[], is_training =False)

        with slim.arg_scope(inception.inception_v1_arg_scope(weight_decay=weight_decay)):
                logits, _ = inception.inception_v1(images, num_classes=2, is_training=False)


        total_output = []
        total_labels = []
        total_images = []

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(train_inceptionV1_bin_dir))
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in range(batch_size):
                print('step: %d/%d' % (i, batch_size))
                o, l , image= sess.run([logits, labels, images[0]])
                o = tf.reduce_sum(o, 0)/float(5)
                total_output.append(o)
                total_labels.append(l[0])
                total_images.append(image)
            coord.request_stop()
            coord.join(threads)


            total_output = tf.stack(total_output,0)
            total_output = tf.nn.softmax(total_output)
            labels = tf.constant(total_labels)
            total_images = sess.run(tf.stack(total_images,0))

            top1_op = tf.nn.in_top_k(total_output, labels, 1)
            top1_acc = sess.run(tf.reduce_mean(tf.cast(top1_op, tf.float32)))
            print(top1_acc)


            top5_op = tf.nn.in_top_k(total_output, labels, 5)
            top5_acc = sess.run(tf.reduce_mean(tf.cast(top5_op, tf.float32)))
            print(top5_acc)

            accuracy1_sum = tf.summary.scalar('top1_accuracy', top1_acc)
            accuracy5_sum = tf.summary.scalar('top5_accuracy', top5_acc)
            images_sum = tf.summary.image("input_images", total_images, batch_size)

            accuracy1, accuracy5, image_batch, step = sess.run([accuracy1_sum,accuracy5_sum,images_sum, global_step])
            writer = tf.summary.FileWriter(eval_inceptionV1_bin_dir)
            writer.add_summary(accuracy1, step)
            writer.add_summary(accuracy5, step)
            writer.add_summary(image_batch)
            #return top1_acc, top5_acc
