from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import math
import urllib.request as urllib2
slim = tf.contrib.slim

from datasets import dataset_utils
from datasets import imagenet
from nets import inception
from nets import vgg
from preprocessing import inception_preprocessing
from preprocessing import vgg_preprocessing
from datasets import plantclef2015_all_labels

from datasets import plantVSnoplant
import matplotlib.pyplot as plt
import sabanci_system
from preprocessing import sabanci_preprocessing

plantVSnoplant_data_dir = 'plantVSnoplant'

test_data_dir= "plantclef2016_test"
train_inceptionV1_dir = 'inception_finetuned'
train_inceptionV1_bin_dir = 'inception_bin_finetuned'
train_vgg16_dir = 'vgg16_finetuned'
eval_inceptionV1_dir = 'inception_evaluation'
eval_vgg16_dir = 'vgg16_evaluation'




tf.app.flags.DEFINE_integer('batch_size', 20, 'The number of samples in each batch.')



FLAGS = tf.app.flags.FLAGS



#run1
weight_decay = 0.0002
start_learning_rate = 0.001
updating_iteration_for_learning_rate =12000
updating_gamma =0.96
momentum = 0.9
num_classes = 1000
batch_size = 20


# preprocessing
num_patches_inception = 9
r_rotations_inception = [10,20]

num_patches_vgg = 5
r_rotations_vgg = [10]


checkpoint_paths = {'inceptionV1':train_inceptionV1_dir, 'vgg16' : train_vgg16_dir}
network_patches = {'inceptionV1':num_patches_inception ,'vgg16':num_patches_vgg }
network_rotations = {'inceptionV1':r_rotations_inception ,'vgg16': r_rotations_vgg}


def get_network_logits(network, images):
  if(network == 'inceptionV1'):
    with slim.arg_scope(inception.inception_v1_arg_scope(weight_decay=weight_decay)):
      logits, _ = inception.inception_v1(images, num_classes=1000, is_training=False)

  elif(network == 'vgg16'):

    with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
      logits, _ = vgg.vgg_16(images, num_classes=1000, is_training=False)

  return logits



output_list = []  # predictions for all images for each network
labels_list = []
output_dict= {}

for network in checkpoint_paths:
  with tf.Graph().as_default():
      tf.logging.set_verbosity(tf.logging.INFO)
      global_step = slim.get_or_create_global_step()

      dataset = plantVSnoplant.get_split('validation_no_plant', plantVSnoplant_data_dir)

      images, labels = sabanci_system.load_batch(dataset, k=network_patches[network], r=network_rotations[network], is_training =False)


      logits = get_network_logits(network, images)

            if network == 'inceptionV1_bin':
                total_output = np.empty([8000,2])

            else:
                total_output = np.empty([8000,1000])

            total_labels = np.empty([8000],dtype = np.int32)



            with tf.Session() as sess:
                coord = tf.train.Coordinator()
                saver = tf. train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_paths[network]))
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                for i in range(8000):
                    print('step: %d/%d' % (i, 8000))
                    o,l = sess.run([logits, labels])
                    o = np.sum(o, 0)/float(5* (network_patches[network] + 2 * len(network_rotations[network]) + 1)) # score level averaging over patches

                    total_output[i] = o
                    total_labels[i] = l[0]


                coord.request_stop()
                coord.join(threads)


            output_dict[network]= total_output
            output_list.append(total_output)
            labels_list.append(total_labels)



# total_count = 8000 # total amount of images
# output_sum = tf.zeros([8000, 1000])

# for output in output_list: # averaging over networks
#   logits = tf.cast(tf.constant(output), dtype = tf.float32)
#   output_sum += logits

# output_sum = output_sum/len(output_list)

# apply softmax function
predictions_V1 = tf.nn.softmax(tf.cast(tf.constant(output_dict['inceptionV1']), dtype = tf.float32))
predictions_vgg = tf.nn.softmax(tf.cast(tf.constant(output_dict['vgg16']), dtype = tf.float32))

# top 1 score index
argmax_V1 = tf.argmax(predictions_V1, 1)
argmax_vgg = tf.argmax(predictions_vgg, 1)


with tf.Session() as sess:
  predictions_V1, argmax_V1, predictions_vgg, argmax_vgg = sess.run([predictions_V1, argmax_V1, predictions_vgg,argmax_vgg])
  predictions_summed = (predictions_V1 + predictions_vgg)/2 # averaging over networks


predictions_filename = os.path.join('/models/slim', 'threshold_data.txt')
seperated_predictions_filename = os.path.join('/models/slim/', 'seperated_data.txt')

with tf.gfile.Open(predictions_filename, 'w') as f:
    with tf.gfile.Open(seperated_predictions_filename, 'w') as g:
        g.write("InceptionV1:Class --- VGG16:Class \n")
        for i in range(len(predictions_V1)):
            max_score_V1 = max(predictions_V1[i,:])
            max_score_vgg = max(predictions_vgg[i,:])
            max_score = max(predictions_summed[i,:])
            f.write("%f," % max_score)
            g.write("%f:%d, %f:%d\n" % (max_score_V1,argmax_V1[i],max_score_vgg, argmax_vgg[i]))
            #for j in range(len(predictions_summed[i,:])):
              #f.write("%f," % predictions_summed[i,j])
              #g.write("%f:%f," % (predictions_V1[i,j],predictions_vgg[i,j]))
            #f.write("\n")
            #g.write("\n")



