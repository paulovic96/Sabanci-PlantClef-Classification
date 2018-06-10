from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from PIL import Image
from datasets import dataset_utils

_NUM_VAlIDATION = 41708
_NUM_PLANT_VALIDATION = 20854
_NUM_NO_PLANT_VALIDATION = 20854
_NUM_TRAINING = 141808
_NUM_SHARDS_TRAINING = 256
_NUM_SHARDS_VALIDATION = 76
_NUM_SHARDS_SPLIT = 38

# Seed for repeatability.
_RANDOM_SEED = 0

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


def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """

  root = os.path.join(dataset_dir, 'plant_vs_noplant')
  directories = []

  for filename in os.listdir(root):
    path = os.path.join(root, filename)
    if os.path.isdir(path):
      directories.append(path)


  photo_filenames= {}
  plant_filenames= []
  no_plant_filenames = []
  class_names = []

  for directory in directories:

    if directory.endswith('noplant'):
        class_names.append('noplant')
        for file in os.listdir(directory):
            path = os.path.join(directory, file)
            no_plant_filenames.append(path)
    else:
        class_names.append('plant')
        for file in os.listdir(directory):
            if file.endswith('.jpg'):
                path = os.path.join(directory, file)
                plant_filenames.append(path)

  photo_filenames['plants'] = plant_filenames
  photo_filenames['no_plants'] = no_plant_filenames

  return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id,num_shards):
    output_filename = 'plantVSnoplant_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, num_shards)
    return os.path.join(dataset_dir, output_filename)







def _convert_dataset(split_name,filenames, class_names_to_ids ,dataset_dir, num_shards):
  assert split_name in ['train', 'validation', 'validation_plant', 'validation_no_plant']
  num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

        for shard_id in range(num_shards):
            output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards)

            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                for i in range(start_ndx, end_ndx):
                    sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(filenames), shard_id))
                    sys.stdout.flush()

                    # Read the filename:
                    image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
                    height, width = image_reader.read_image_dims(sess, image_data)
                    # class id:
                    class_name = os.path.basename(os.path.dirname(filenames[i]))
                    class_id = class_names_to_ids[class_name]

                    example = dataset_utils.image_to_tfexample(image_data, b'jpg', height, width, class_id)
                    tfrecord_writer.write(example.SerializeToString())


    sys.stdout.write('\n')
    sys.stdout.flush()

def _clean_up_temporary_files(dataset_dir):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  tf.gfile.Remove(filepath)

  tmp_dir = os.path.join(dataset_dir, 'plant_vs_noplant')
  tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation', 'validation_plant', 'validation_no_plant']:
    for num_shards in[256, 76, 38, 38]:
        for shard_id in range(num_shards):
            output_filename = _get_dataset_filename(
                dataset_dir, split_name, shard_id, num_shards)
            if not tf.gfile.Exists(output_filename):
                return False
  return True


def run(dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  #dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
  photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))
  plant_filenames = photo_filenames['plants']
  no_plant_filenames = photo_filenames['no_plants']


  # check no_plant_filenames for corrupted images
  corrupted_images = 0
  no_plant_save_filenames = []
  print(len(no_plant_filenames))


  # Divide into train and test:

  random.seed(_RANDOM_SEED)
  random.shuffle(plant_filenames)
  random.shuffle(no_plant_save_filenames)

  training_filenames = plant_filenames[_NUM_PLANT_VALIDATION:]
  training_filenames.extend(no_plant_save_filenames[_NUM_NO_PLANT_VALIDATION:91758])


  random.seed(_RANDOM_SEED)
  random.shuffle(training_filenames)


  validation_plant_filenames = plant_filenames[:_NUM_PLANT_VALIDATION]
  validation_no_plant_filenames = no_plant_save_filenames[:_NUM_NO_PLANT_VALIDATION]


  validation_filenames = validation_plant_filenames
  validation_filenames.extend(validation_no_plant_filenames)


  random.seed(_RANDOM_SEED)
  random.shuffle(validation_filenames)



  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, class_names_to_ids,
                   dataset_dir,_NUM_SHARDS_TRAINING)

  _convert_dataset('validation', validation_filenames, class_names_to_ids,
                   dataset_dir, _NUM_SHARDS_VALIDATION)


  # Second, convert the validation sets for plant and no_plant separately
  #_convert_dataset('validation_plant', validation_plant_filenames, class_names_to_ids,
                   #dataset_dir, _NUM_SHARDS_SPLIT)
  #_convert_dataset('validation_no_plant', validation_no_plant_filenames, class_names_to_ids,
                   #dataset_dir, _NUM_SHARDS_SPLIT)


  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  #_clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the PlantVsNoplant dataset!')
