from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN_Training = 'training_data_2015_%s_*.tfrecord'
_FILE_PATTERN_TEST = 'test_data_2015_%s_*.tfrecord'
#
SPLITS_TO_SIZES = {'train': 70904, 'validation': 20854, 'test':21446, 'train_whole': 102777, 'validation_whole': 10427}

_NUM_CLASSES = 1000

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label_species': 'A single integer between 0 and 999 ',
    'label_genus': 'A single integer between 0 and 515',
    'label_family': 'A single integer between 0 and 123',
    'label_organ': 'A single integer between 0 and 6'
}


def get_split(split_name,dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  if split_name not in SPLITS_TO_SIZES :
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN_Training
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),'image/class/label_species': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),'image/class/label_genus': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),'image/class/label_family': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),'image/class/label_organ': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))}

  items_to_handlers = {'image': slim.tfexample_decoder.Image(),'label_species': slim.tfexample_decoder.Tensor('image/class/label_species'),'label_genus': slim.tfexample_decoder.Tensor('image/class/label_genus'),'label_family': slim.tfexample_decoder.Tensor('image/class/label_family'),'label_organ': slim.tfexample_decoder.Tensor('image/class/label_organ') }

  decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)
