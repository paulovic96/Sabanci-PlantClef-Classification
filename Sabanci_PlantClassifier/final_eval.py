from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import urllib.request as urllib2
slim = tf.contrib.slim

from datasets import dataset_utils
from datasets import imagenet
from nets import inception
from nets import vgg
from preprocessing import inception_preprocessing
from preprocessing import vgg_preprocessing
from datasets import plantclef2015
from datasets import plantclef2016
from datasets import flowers
import matplotlib.pyplot as plt
import sabanci_system
from preprocessing import sabanci_preprocessing

import math

test_data_dir= "/plantclef2016_test"
train_inceptionV1_dir = '/inception_finetuned'
train_inceptionV1_bin_dir = '/inception_bin_finetuned'
train_vgg16_dir = '/vgg16_finetuned'
eval_inceptionV1_dir = '/inception_evaluation'
eval_vgg16_dir = '/vgg16_evaluation'

final_eval_dir = '/final_evaluation'

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




FLAGS = tf.app.flags.FLAGS

checkpoint_paths = {'inceptionV1':train_inceptionV1_dir, 'vgg16' : train_vgg16_dir, 'inceptionV1_bin': train_inceptionV1_bin_dir}
network_patches = {'inceptionV1':num_patches_inception ,'vgg16':num_patches_vgg , 'inceptionV1_bin':0}
network_rotations = {'inceptionV1':r_rotations_inception ,'vgg16': r_rotations_vgg, 'inceptionV1_bin':[]}


def get_network_logits(network, images):
  if(network == 'inceptionV1'):
    with slim.arg_scope(inception.inception_v1_arg_scope(weight_decay=weight_decay)):
      logits, _ = inception.inception_v1(images, num_classes=1000, is_training=False)

  elif(network == 'vgg16'):
    with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
      logits, _ = vgg.vgg_16(images, num_classes=1000, is_training=False)

  elif(network == 'inceptionV1_bin'):
     with slim.arg_scope(inception.inception_v1_arg_scope(weight_decay=weight_decay)):
      logits, _ = inception.inception_v1(images, num_classes=2, is_training=False)

  return logits




output_dict = {}
output_list = []
media_list = []
bin_outputs = np.empty([8000,2])


for network in checkpoint_paths:
    print(network)
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        global_step = slim.get_or_create_global_step()

        dataset = plantclef2016.get_split('test', test_data_dir)

        images, image_ids = sabanci_system.load_batch(dataset, k=network_patches[network], r=network_rotations[network], is_training =False, is_testing=True)


        logits = get_network_logits(network, images)


        if network == 'inceptionV1_bin':
            total_output = np.empty([8000,2])

        else:
            total_output = np.empty([8000,1000])

        total_image_ids = np.empty([8000],dtype = np.int32)



        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            saver = tf. train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_paths[network]))
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(8000):
                print('step: %d/%d' % (i, 8000))
                o,l = sess.run([logits, image_ids])
                o = np.sum(o, 0)/float(5* (network_patches[network] + 2 * len(network_rotations[network]) + 1))

                total_output[i] = o
                total_image_ids[i] = l[0]


            coord.request_stop()
            coord.join(threads)

        if(network == 'inceptionV1_bin'):
            bin_outputs = total_output
        else:
            output_dict[network]= total_output
            output_list.append(total_output)
        media_list.append(total_image_ids)

total_count = 8000



# Network averaging
output_sum = tf.zeros([8000, 1000])
for output in output_list:
    logits = tf.cast(tf.constant(output), dtype = tf.float32)
    output_sum += logits

output_sum = output_sum/len(output_list)

# apply softmax function (Predicitions)
predictions_plant= tf.nn.softmax(output_sum)
predictions_bin = tf.nn.softmax(bin_outputs)
image_ids = media_list[0]
plant = tf.argmax(predictions_bin,1)


with tf.Session() as sess:
    predictions_plant,predictions_bin,plant = sess.run([predictions_plant,predictions_bin,plant])



run_file = 'sabanci_run1.txt'

labels_to_class_names = dataset_utils.read_label_file('/plantclef2015')

labels_filename = os.path.join(final_eval_dir, run_file)


rejected_noplants = 0
rejected_plants = 0
with tf.gfile.Open(labels_filename, 'w') as f:
    for i in range(len(image_ids)):

        if plant[i]: # 1=plant 0=nonplant
            if max(predictions_plant[i,:]) >= 0.4: # threshold
                current_image = predictions_plant[i,:]
                for j in range(len(current_image)):
                    f.write('%s;%s;%f\n' % (image_ids[i], labels_to_class_names[j], predictions_plant[i,j]))
            else:
                rejected_plants += 1

        else:

            rejected_noplants += 1


print("Finished testing: Rejected %d images as nonplant objects and %d images as unkown plant!!" % (rejected_noplants, rejected_plants))





