from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

from lxml import etree

import tensorflow as tf

from datasets import dataset_utils

_DATA_SPLIT = {'training_data_2014': 'http://otmedia.lirmm.fr/LifeCLEF/LifeCLEF2014/TrainPackages/PlantCLEF2014trainAllInOne.tar',
'training_data_2015' : 'http://otmedia.lirmm.fr/LifeCLEF/PlantCLEF2015/Packages/TrainingPackage/PlantCLEF2015TrainingData.tar.gz',
'test_data_2015' : 'http://otmedia.lirmm.fr/LifeCLEF/PlantCLEF2015/Packages/TestPackage/PlantCLEF2015TestDataWithAnnotations.tar.gz',
'whole_data_2015' : "Training+Test+Validation/2",
'test_data_2016' : "http://otmedia.lirmm.fr/LifeCLEF/PlantCLEF2016/PlantCLEF2016Test.tar.gz"}


# The number of images in the validation set.
#_NUM_VALIDATION = #2014
#_NUM_VALIDATION = 20854 #2015
#_NUM_VALIDATION =  10427

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
# training data
#_NUM_SHARDS = 128
# test 2015
#_NUM_SHARDS = 42
# whole data
#_NUM_SHARDS = 201
# test 2016
_NUM_SHARDS = 18

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string) #image data
    self._decode_jpeg = tf.image.decode_image(self._decode_jpeg_data, channels=3) #decoded jpeg data

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,feed_dict={self._decode_jpeg_data: image_data}) #feed image data to decode
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _get_filenames_with_ids_and_class_names(dataset_dir):
  root = os.path.join(dataset_dir, 'PlantCLEF2016Test')
  filenames = []	# files
  media_ids = []
  metadata = []	# xml dat

  for file in os.listdir(root):
    path = os.path.join(root, file)
    if file.endswith('.xml'):
      metadata.append(path)
    elif file.endswith('.jpg'):
      filenames.append(path)

  filenames = sorted(filenames)
  metadata = sorted(metadata)

  assert len(filenames) == len(metadata)

  for i in range(len(metadata)):
    class_data = etree.parse(metadata[i])
    media_id = class_data.findtext('MediaId')
    media_ids.append(int(media_id))


  files =  filenames
  media_dict = dict(zip(filenames, media_ids))
  return files, media_dict                    # files to class_ids, class_ids to class_names


def _get_dataset_filename(dataset_dir, dataset_name, split_name, shard_id):
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (dataset_name,
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name,filenames, media_dict, dataset_dir, dataset_name):
  assert split_name in ['train', 'validation', 'test', 'train_whole', 'validation_whole']
  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(dataset_dir, dataset_name, split_name, shard_id)

            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                for i in range(start_ndx, end_ndx):
                    sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(filenames), shard_id))
                    sys.stdout.flush()

                    # Read the filename:
                    image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
                    height, width = image_reader.read_image_dims(sess, image_data)

                    # media id:

                    media_id = media_dict[filenames[i]]

                    example = dataset_utils.image_to_tfexample(image_data, b'jpg', height, width,media_id)
                    tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()

def _clean_up_temporary_files(dataset_dir,dataset_name):
  """Removes temporary files used to create the dataset.
  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_SPLIT[dataset_name].split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  if os.path.exists(filepath):
    tf.gfile.Remove(filepath)
    tmp_dir = os.path.join(dataset_dir, 'PlantCLEF2016Test') # Testset 2016
    tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir,dataset_name):
  for split_name in ['train', 'validation', 'test', 'train_whole', 'validation_whole']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(dataset_dir, dataset_name, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def run(dataset_dir, dataset_name):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """


  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir, dataset_name):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  if dataset_name not in _DATA_SPLIT:
    raise ValueError('Dataset name %s was not recognized.' % dataset_name)

  dataset_utils.download_and_uncompress_tarball_with_transparent_compression(_DATA_SPLIT[dataset_name], dataset_dir)
  photo_filenames, media_dict = _get_filenames_with_ids_and_class_names(dataset_dir)


  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)

  test_filenames = photo_filenames


  _convert_dataset('test', test_filenames, media_dict,
                   dataset_dir, dataset_name)


  print('\nFinished converting the Plantclef dataset!')

