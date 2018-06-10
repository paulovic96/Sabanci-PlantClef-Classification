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
from datasets import flowers
from datasets import plantVSnoplant
import matplotlib.pyplot as plt
from preprocessing import sabanci_preprocessing


import tf_cnnvis.tf_cnnvis
import scipy.misc

import top9_activations

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


# Training Data
train_inceptionV1_dir = 'inception_finetuned'
train_vgg16_dir = 'vgg16_finetuned'


def visualize_cnnvis_max(network,layer):
    """ Get feature Map visualization and deconvolution
    Args:
        network: Network of intererst
        layer: layer of interest

    """
    activations = top9_activations.get_activations_for_top9(network,layer) # get top9 activations
    print("\nAll activations found and saved.")


    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)

    if network == 'inceptionV1':
        model_fn =  'inception_finetuned/frozen_model.pb' # frozen graph

    elif network == 'vgg16':
        model_fn =  'vgg16_finetuned/frozen_model.pb' # frozen graph


    with tf.gfile.FastGFile(model_fn, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())


    t_input = tf.placeholder(np.float32, name='batch') # define the input tensor

    tf.import_graph_def(graph_def, input_map={'batch':t_input})


    for key in activations.keys(): # over filters
        images, values, labels, images_raw = activations[key]
        for i in range(9): # over top 9 images
            value =values[i]
            image = images[i]
            label = labels[i]
            image_raw = images_raw[i]

            im = np.expand_dims(image, axis = 0)


        # specify target layer to visualize depending on name in graph
        # specify filename to store the visualizations
            if network == 'inceptionV1':
                if "Conv" in layer:
                    target = "import/InceptionV1/InceptionV1/%s/convolution" % layer
                    filename = 'import_inceptionv1_inceptionv1_%s_convolution' % layer.lower()
                elif "Max" in layer:
                    target = "import/InceptionV1/InceptionV1/%s/MaxPool" % layer
                    filename = 'import_inceptionv1_inceptionv1_%s_maxpool' % layer.lower()
                elif "Logits" in layer:
                    target = "import/InceptionV1/Logits/SpatialSqueeze"
                    filename = 'import_inceptionv1_logits_spatialsqueeze'
                else:
                    target = "import/InceptionV1/InceptionV1/%s/concat" % layer
                    filename = 'import_inceptionv1_inceptionv1_%s_concat' % layer.lower()

            elif network == 'vgg16':
                if "conv" in layer:
                    target = "import/%s/convolution" % layer
                    filename = "import_%s_convolution" % layer.lower().replace('/','_')
                elif "pool" in layer:
                    target = "import/%s/MaxPool" % layer
                    filename = "import_%s_maxpool" % layer.lower().replace('/','_')
                elif "fc8" in layer:
                    target = "import/%s/squeezed" % layer
                    filename = "import_%s_squeezed" % layer.lower().replace('/','_')
                else:
                    target = "import/%s/convolution" % layer
                    filename = "import_%s_convolution" % layer.lower().replace('/','_')


            # feature map visualization
            tf_cnnvis.tf_cnnvis.activation_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = {t_input : im}, layers = target, path_outdir = "./my_Outputs_%s" % network)

            # deconvolution
            tf_cnnvis.tf_cnnvis.deconv_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = {t_input : im}, layers = target, path_outdir = "./my_Outputs_%s" % network) #import/resnet_v2_50/pool5 #layers

            # store coressponding original images
            plt.figure()
            plt.imshow(image_raw.astype('uint8'))
            plt.title('Ground Truth: [%d]' % label)
            plt.axis('off')
            plt.savefig('my_Outputs_%s/%s/filter_%d_image_%d.png' % (network,filename,key,i))

    print("DONE")


