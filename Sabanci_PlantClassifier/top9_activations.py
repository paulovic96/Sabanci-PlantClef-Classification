

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import os
import urllib.request as urllib2
import math
from operator import itemgetter
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

inceptionV1_activations_dir ='inceptionV1_activations'
vgg16_activations_dir = 'vgg16_activations'


checkpoint_paths = {'inceptionV1':train_inceptionV1_dir, 'vgg16' : train_vgg16_dir}
activation_paths = {'inceptionV1':inceptionV1_activations_dir, 'vgg16' : vgg16_activations_dir}


inceptionV1_endpoints= ['Mixed_4e', 'Conv2d_1a_7x7', 'Conv2d_2c_3x3', 'Mixed_3b', 'Mixed_3c', 'MaxPool_3a_3x3', 'Mixed_4d', 'Mixed_4c', 'Logits', 'MaxPool_4a_3x3', 'Conv2d_2b_1x1', 'MaxPool_2a_3x3', 'Predictions', 'Mixed_5b', 'Mixed_4f', 'Mixed_4b', 'Mixed_5c', 'MaxPool_5a_2x2']
vgg16_endpoints = ['vgg_16/conv1/conv1_1', 'vgg_16/conv1/conv1_2', 'vgg_16/pool1', 'vgg_16/conv2/conv2_1', 'vgg_16/conv2/conv2_2', 'vgg_16/pool2', 'vgg_16/conv3/conv3_1', 'vgg_16/conv3/conv3_2', 'vgg_16/conv3/conv3_3', 'vgg_16/pool3', 'vgg_16/conv4/conv4_1', 'vgg_16/conv4/conv4_2', 'vgg_16/conv4/conv4_3', 'vgg_16/pool4', 'vgg_16/conv5/conv5_1', 'vgg_16/conv5/conv5_2', 'vgg_16/conv5/conv5_3', 'vgg_16/pool5', 'vgg_16/fc6', 'vgg_16/fc7', 'vgg_16/fc8']

network_endpoints = {'inceptionV1':inceptionV1_endpoints, 'vgg16' : vgg16_endpoints}


def get_network_logits_and_endpoints(network, images):
  if(network == 'inceptionV1'):
    with slim.arg_scope(inception.inception_v1_arg_scope(weight_decay=weight_decay)):
      logits, endpoints = inception.inception_v1(images, num_classes=1000, is_training=False)

  elif(network == 'vgg16'):

    with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
      logits, endpoints = vgg.vgg_16(images, num_classes=1000, is_training=False)

  return logits,endpoints






def layer_top9_helper(network,iterations,burnin = 0,classes="species"):
    """ get the top 9 activity for all filters in all layer endpoints
    Args:
        network: the network of interest
        iterations: number images shown to the network
        burnin: "burn" the images already seen
    """
  with tf.Graph().as_default():

    # reading in test images
    dataset = plantclef2015_all_labels.get_split('train', plant_data_dir)
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32,common_queue_min=8, shuffle = False)

    image_raw, label = data_provider.get(['image', 'label_%s' % classes])

    image = tf.image.per_image_standardization(image_raw)
    image = tf.image.resize_images(image, [224, 224])

    images, labels = tf.train.batch([image, label],batch_size=1, shapes = [tf.TensorShape([tf.Dimension(224), tf.Dimension(224), tf.Dimension(3)]), tf.TensorShape([])],num_threads=1, capacity=2 * 20)


    _,endpoints = get_network_logits_and_endpoints(network,images)

    with tf.Session() as sess:

        layer_dict={}
        for endpoint in endpoints:
            layer = endpoints[endpoint] #layer activation
            layer_dict[endpoint] = np.zeros([layer.get_shape().as_list()[-1],9,3]) # new entry with a np.arry containing  for all filters, for the top9 activating images image_nr, current_filter_summed_activation, label

        coord = tf.train.Coordinator()
        saver = tf. train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_paths[network]))
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(burnin): # burn images
                print('burnin: %d/%d' % (i,burnin))
                sess.run(images)

        for i in range(iterations):#dataset.num_samples):
            print('step: %d/%d' % (i+burnin,dataset.num_samples))
            nets, label = sess.run([endpoints,labels[0]])
            image_nr = i


            for endpoint in nets: # for each layer
                net = nets[endpoint]
                layer_top9 = layer_dict[endpoint]
                for j in range(net.shape[-1]): # for all filters in layer
                    current_filter = net[...,j]
                    current_filter_absolute_activations = np.absolute(current_filter) # summed activation
                    current_filter_summed_activation = np.sum(current_filter_absolute_activations)
                    #current_filter_max_across = np.amax(current_filter) # max activation

                    filter_top9 = layer_top9[j,:,:]
                    filter_top9 = np.asarray(sorted(filter_top9, key=itemgetter(1))) # sort by summed activation

                    if current_filter_summed_activation >= filter_top9[0,1]: # bigger activation found
                        filter_top9[0,:] = [image_nr, current_filter_summed_activation, label]
                        layer_top9[j,:,:] = np.asarray(sorted(filter_top9, key=itemgetter(1))) # sort by summed activation
                layer_dict[endpoint]=layer_top9
        sess.close()
        return layer_dict




def network_layer_top9(network):
    """ get the top 9 activity for all filters in all layer endpoints
    Args:
        network: the network of interest
    """
    num_iterations = 70904//500 # num validation images in 500 batches
    rest = 70904%500

    global_top9 = layer_top9_helper(network,500) # save top9 activations across all validation images
    for i in range(1,num_iterations):
        batch_top9 = layer_top9_helper(network,500,burnin = i*500) # top9 activations across 500 images
        for layer in batch_top9: # compare each layer
            current_layer_top9 = batch_top9[layer]
            global_layer_top9 = global_top9[layer]
            for j in range(len(current_layer_top9)): # compare each filter
                    current_filter_top9 = current_layer_top9[j,:,:] # current filter activation
                    global_filter_top9 = global_layer_top9[j,:,:] # global filter activation
                    global_filter_top9 = np.concatenate((global_filter_top9,current_filter_top9), axis=0) # concat to find top 9 across both
                    global_filter_top9 = np.asarray(sorted(global_filter_top9, key=itemgetter(1))) # sort by activation
                    global_layer_top9[j,:,:] = global_filter_top9[-9:] # get top 9
            global_top9[layer] = global_layer_top9


    # compare for remaining images
    batch_top9 = layer_top9_helper(network,rest,burnin = num_iterations*500)
    for layer in batch_top9:
        current_layer_top9 = batch_top9[layer]
        global_layer_top9 = global_top9[layer]
        for j in range(len(current_layer_top9)): # over filter
                current_filter_top9 = current_layer_top9[j,:,:]
                for k in range(len(current_filter_top9)): # over top 9
                    if current_filter_top9[k,1] >= global_layer_top9[j,k,1]:
                        global_layer_top9[j,k,:] = current_filter_top9[k,:]
                        global_layer_top9[j,:,:] = np.asarray(sorted(global_layer_top9[j,:,:], key=itemgetter(1)))
        global_top9[layer] = global_layer_top9


    # write down activations for each layer
    for result in global_top9:
        if network == 'vgg16':
            layer_file = 'top9_activation_%s.txt' % result.replace('/','_')
        else:
            layer_file = 'top9_activation_%s.txt' % result
        layer_filename = os.path.join(activation_paths[network], layer_file)

        global_layer_top9 = global_top9[result]
        with tf.gfile.Open(layer_filename, 'w') as f:
            f.write('%d\n' % len(global_layer_top9))
            f.write('Idx in Data  Activity  Label \n')
            for i in range(len(global_layer_top9)):
                current_filter = global_layer_top9[i,:,:]
                f.write('Filter_%d\n' % i)
                for j in range(len(current_filter)):
                    f.write('%d;%f;%d\n' % (current_filter[j,0], current_filter[j,1], current_filter[j,2]))
                f.write('\n')



def get_activations_for_top9_helper(network,layer, filter_num, image_list, classes='species'):
    """ get the top 9 activations for all filters in all layer endpoints from files
    Args:
        network: the network of interest
        layer: the layer of interest
        filter_num: filter of interest
        image_list: idx of top9 images
        classes: class of interest (one of 'species', 'genus', 'family', 'organ')
    returns:
        filter_top9: top9 filter activations
        filter_top9_img: top9 images
        original_top9_img: filter_top9 original image
    """
    with tf.Graph().as_default():
        dataset = plantclef2015_all_labels.get_split('train', plant_data_dir)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32,common_queue_min=8, shuffle = False)

        image_raw, label = data_provider.get(['image', 'label_%s' % classes])

        image_raw = tf.image.resize_images(image_raw, [224, 224])
        image = tf.image.per_image_standardization(image_raw)

        images, labels, images_raw = tf.train.batch([image,label, image_raw],batch_size=1, shapes = [tf.TensorShape([tf.Dimension(224), tf.Dimension(224), tf.Dimension(3)]), tf.TensorShape([]),tf.TensorShape([tf.Dimension(224), tf.Dimension(224), tf.Dimension(3)])],num_threads=1, capacity=2 * 20)

        _, endpoints = get_network_logits_and_endpoints(network, images)

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            saver = tf. train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_paths[network]))
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            filter_top9 = np.empty(9, dtype=object)
            filter_top9_img = np.empty(9, dtype=object)
            original_top9_img = np.empty(9, dtype=object)
            sorted_image_idx = sorted(range(len(image_list)), key=lambda i: image_list[i]) # sort image idx to get first image first independent of position in top9 activations

            for i in range(dataset.num_samples):
                if not sorted_image_idx:
                    print('Found Top 9 activating images for filter:%d in layer:%s' % (filter_num, layer))
                    coord.request_stop()
                    coord.join(threads)
                    return filter_top9, filter_top9_img, original_top9_img

                else:
                    if i == image_list[sorted_image_idx[0]]:
                        activation,image, raw_image = sess.run([endpoints[layer][0],images[0],images_raw[0]])
                        activation_filter = activation[...,filter_num]
                        filter_top9[sorted_image_idx[0]] = activation_filter
                        filter_top9_img[sorted_image_idx[0]] = image
                        original_top9_img[sorted_image_idx[0]] = raw_image
                        sorted_image_idx = sorted_image_idx[1:]
                    else:
                        sess.run(images)



def get_activations_for_top9(network,layer):
     """ get the top 9 activations for random choosen filters in all layer endpoints from files
    Args:
        network: the network of interest
        layer: the layer of interest
    returns:
         layer_activations:
    """
    if network == "inceptionV1":
        layer_file = 'top9_activation_%s.txt' % layer
    elif network == "vgg16":
        layer_file = 'top9_activation_%s.txt' % layer.replace('/','_')
    layer_filename = os.path.join(activation_paths[network], layer_file)
    with tf.gfile.Open(layer_filename, 'r') as f:
        lines = f.readlines()
        num_filters = int(lines[0])

        lines = [line for line in lines[2:] if "Filter" not in line and line != "\n"] # skip head

    layer_activations = {}
    filter_list={}
    for i in range(num_filters):
        current_filter_image_list=[]
        current_filter_label_list=[]
        current_filter_strings = lines[i*9:(i*9)+9] # top9 filter activations

        for j in range(len(current_filter_strings)):
            current_filter_strings[j]=current_filter_strings[j].replace("\n", '')
            split = current_filter_strings[j].split(";")
            split = [i for i in split if i != '']
            split = list(map(float, split))
            current_filter_image_list.append(split[0])
            current_filter_label_list.append(int(split[2]))

        filter_list[i] =[current_filter_image_list,current_filter_label_list]


    keys = list(filter_list.keys())
    # randomly choose filters from each depth of a mixed module. In order to get one for all operations used in the module
    if layer == 'Mixed_3b': #Mixed_3b
        print(layer)
        x1_filter = np.random.choice(keys[0:64],1,replace=False)
        x3_filter = np.random.choice(keys[64:192],1, replace=False)
        x5_filter = np.random.choice(keys[192:224],1, replace=False)
        max_pool_filter = np.random.choice(keys[224:256],1,replace=False)
        keys = np.concatenate((x1_filter,x3_filter,x5_filter,max_pool_filter))

    elif layer == 'Mixed_3c':
        x1_filter = np.random.choice(keys[0:128],1,replace=False)
        x3_filter = np.random.choice(keys[128:320],1, replace=False)
        x5_filter = np.random.choice(keys[320:416],1, replace=False)
        max_pool_filter = np.random.choice(keys[416:480], 1,replace=False)
        keys = np.concatenate((x1_filter,x3_filter,x5_filter,max_pool_filter))

    elif layer == 'Mixed_4b':
        x1_filter = np.random.choice(keys[0:192],1,replace=False)
        x3_filter = np.random.choice(keys[192:400],1, replace=False)
        x5_filter = np.random.choice(keys[400:448],1, replace=False)
        max_pool_filter = np.random.choice(keys[448:512], 1,replace=False)
        keys = np.concatenate((x1_filter,x3_filter,x5_filter,max_pool_filter))


    elif layer == 'Mixed_4c':
        x1_filter = np.random.choice(keys[0:160],1,replace=False)
        x3_filter = np.random.choice(keys[160:384],1, replace=False)
        x5_filter = np.random.choice(keys[384:448],1, replace=False)
        max_pool_filter = np.random.choice(keys[448:512], 1,replace=False)
        keys = np.concatenate((x1_filter,x3_filter,x5_filter,max_pool_filter))


    elif layer == 'Mixed_4d':
        x1_filter = np.random.choice(keys[0:128],1,replace=False)
        x3_filter = np.random.choice(keys[128:384],1, replace=False)
        x5_filter = np.random.choice(keys[384:448],1, replace=False)
        max_pool_filter = np.random.choice(keys[448:512], 1,replace=False)
        keys = np.concatenate((x1_filter,x3_filter,x5_filter,max_pool_filter))

    elif layer == 'Mixed_4e':
        x1_filter = np.random.choice(keys[0:112],1,replace=False)
        x3_filter = np.random.choice(keys[112:400],1, replace=False)
        x5_filter = np.random.choice(keys[400:464],1, replace=False)
        max_pool_filter = np.random.choice(keys[464:528], 1,replace=False)
        keys = np.concatenate((x1_filter,x3_filter,x5_filter,max_pool_filter))

    elif layer == 'Mixed_4f':
        x1_filter = np.random.choice(keys[0:256],1,replace=False)
        x3_filter = np.random.choice(keys[256:576],1, replace=False)
        x5_filter = np.random.choice(keys[576:704],1, replace=False)
        max_pool_filter = np.random.choice(keys[704:832], 1,replace=False)
        keys = np.concatenate((x1_filter,x3_filter,x5_filter,max_pool_filter))

    elif layer == 'Mixed_5b':
        x1_filter = np.random.choice(keys[0:256],1,replace=False)
        x3_filter = np.random.choice(keys[256:576],1, replace=False)
        x5_filter = np.random.choice(keys[576:704],1, replace=False)
        max_pool_filter = np.random.choice(keys[704:832], 1,replace=False)
        keys = np.concatenate((x1_filter,x3_filter,x5_filter,max_pool_filter))

    elif layer == 'Mixed_5c':
        x1_filter = np.random.choice(keys[0:384],1,replace=False)
        x3_filter = np.random.choice(keys[384:768],1, replace=False)
        x5_filter = np.random.choice(keys[768:896],1, replace=False)
        max_pool_filter = np.random.choice(keys[896:1024], 1,replace=False)
        keys = np.concatenate((x1_filter,x3_filter,x5_filter,max_pool_filter))

    elif layer == 'Logits' or 'fc8' in layer:
        keys = np.random.choice(keys, 4, replace=False)

    else:
        keys = np.random.choice(keys, 2, replace=False)

    for key in keys:

        top9, top9_img, top_9_img_raw = get_activations_for_top9_helper(network,layer,key, filter_list[key][0])
        layer_activations[key] = [top9_img,top9,filter_list[key][1],top_9_img_raw] #layer activation for filter i in ascending order

    return layer_activations

