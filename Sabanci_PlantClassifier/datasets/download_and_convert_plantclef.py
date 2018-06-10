from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

from lxml import etree

import tensorflow as tf

#from datasets import dataset_utils
import dataset_utils_plantclef

plant_data_dir = 'plantclef2015_all_labels'


_DATA_SPLIT = {'training_data_2014': 'http://otmedia.lirmm.fr/LifeCLEF/LifeCLEF2014/TrainPackages/PlantCLEF2014trainAllInOne.tar',
'training_data_2015' : 'http://otmedia.lirmm.fr/LifeCLEF/PlantCLEF2015/Packages/TrainingPackage/PlantCLEF2015TrainingData.tar.gz',
'test_data_2015' : 'http://otmedia.lirmm.fr/LifeCLEF/PlantCLEF2015/Packages/TestPackage/PlantCLEF2015TestDataWithAnnotations.tar.gz',
'whole_data_2015' : "Training+Test+Validation/2"
'test_data_2016 : "http://otmedia.lirmm.fr/LifeCLEF/PlantCLEF2016/PlantCLEF2016Test.tar.gz'}


# The number of images in the validation set.
_NUM_VALIDATION = 20854 # 2015
#_NUM_VALIDATION =  10427 #

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
# training data
_NUM_SHARDS = 168
# test 2015
#_NUM_SHARDS = 42
# whole data
#_NUM_SHARDS = 201
# test 2016
#_NUM_SHARDS = 18

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
  """
    Args:
      dataset_dir: the location with downloaded material

    returns:
      file_dicts_*: dictionary with filename and coressponding metadata
  """
  root = os.path.join(dataset_dir, 'train')
  #root = os.path.join(dataset_dir, 'PlantCLEF2015TestDataWithAnnotations')
  #root = os.path.join(dataset_dir, 'PlantCLEF2016Test')
  filenames = []	# files
  class_species = []
  class_familys = []
  class_geni = []
  class_ids = []
  class_contents = []
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
    class_family = class_data.findtext('Family')
    class_genus = class_data.findtext('Genus')
    class_spec = class_data.findtext('Species')
    class_id = class_data.findtext('ClassId')
    class_content = class_data.findtext('Content')

    class_species.append(class_spec)
    class_geni.append(class_genus)
    class_familys.append(class_family)
    class_ids.append(class_id)
    class_contents.append(class_content)

  file_dict_species = dict(zip(filenames,class_species))
  file_dict_geni =  dict(zip(filenames,class_geni))
  file_dict_familys = dict(zip(filenames,class_familys))
  file_dict_ids =  dict(zip(filenames,class_ids))
  file_dict_contents =  dict(zip(filenames,class_contents))
  return file_dict_species,file_dict_geni, file_dict_familys, file_dict_ids, file_dict_contents # return dictionarys filenames:metadata


def _get_dataset_filename(dataset_dir, dataset_name, split_name, shard_id):
  """
    Args:
      dataset_dir: the location with downloaded material,
      dataset_name: dataset name,
      split_name: training or validation or test
      shard_id: the current shard
    returns: the dataset filename
  """
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (dataset_name,
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name,filenames, file_dict_geni, file_dict_familys,file_dict_ids, file_dict_contents, class_id_to_label, genus_to_label, family_to_label, organ_to_label, dataset_dir, dataset_name):
  """
    Args:
      split_name: training or validation or test,
      filnames: a list of image filenames,
      file_dict_geni: dictionary mapping from filename to genus
      file_dict_familys: dictionary mapping from filename to family
      file_dict_ids: dictionary mapping from filename to class_id
      file_dict_contetns: dictionary mapping from filename to organs
      *_to_label: dictionary mapping from metadata to label
      dataset_dir: the location with downloaded material,
      dataset_name: dataset name,

      shard_id: the current shard
    returns: the dataset filename
  """
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

                    # class id:
                    class_id = file_dict_ids[filenames[i]]
                    genus = file_dict_geni[filenames[i]]
                    family = file_dict_familys[filenames[i]]
                    organ = file_dict_contents[filenames[i]]


                    species_id = class_id_to_label[class_id]
                    genus_id = genus_to_label[genus]
                    family_id = family_to_label[family]
                    organ_id = organ_to_label[organ]

                    example = dataset_utils.image_to_tfexample(image_data, b'jpg', height, width, species_id,genus_id,family_id,organ_id)
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
    tmp_dir = os.path.join(dataset_dir, 'train') # Trainingset
    #tmp_dir = os.path.join(dataset_dir, 'PlantCLEF2015TestDataWithAnnotations') # Testset
    #tmp_dir = os.path.join(dataset_dir, 'PlantCLEF2016Test') # Testset 2016
    tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir,dataset_name):
  for split_name in ['train', 'validation', 'test', 'train_whole', 'validation_whole', 'whole']:
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

  class_id_to_genus = dataset_utils.read_label_file(plant_data_dir, filename='class_id_to_genus.txt')
  genus_to_label = dataset_utils.read_label_file(plant_data_dir, filename='label_to_genus.txt')
  genus_to_label = {v.replace("\r",""):k for k,v in genus_to_label.items()}

  label_to_class_id = dataset_utils.read_label_file(plant_data_dir, filename='labels.txt')
  class_id_to_label = {v:k for k,v in label_to_class_id.items()}


  class_id_to_family = dataset_utils.read_label_file(plant_data_dir, filename='class_id_to_family.txt')
  family_to_label = dataset_utils.read_label_file(plant_data_dir, filename='label_to_family.txt')
  family_to_label = {v.replace("\r",""):k for k,v in family_to_label.items()}

  label_to_content = dataset_utils.read_label_file(plant_data_dir, filename='label_to_content.txt')
  content_to_label = {v:k for k,v in label_to_content.items()}




  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir, dataset_name):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  if dataset_name not in _DATA_SPLIT:
    raise ValueError('Dataset name %s was not recognized.' % dataset_name)

  dataset_utils.download_and_uncompress_tarball_with_transparent_compression(_DATA_SPLIT[dataset_name], dataset_dir)
  file_dict_species, file_dict_geni, file_dict_familys, file_dict_ids, file_dict_contents = _get_filenames_with_ids_and_class_names(dataset_dir)

  class_names_dict_class_ids = dict.fromkeys(file_dict_class_ids.values())
  class_id_to_labels = dict(zip(list(class_names_dict_class_ids.keys()), range(len(class_names_dict_class_ids))))

  class_names_dict_geni = dict.fromkeys(file_dict_geni.values())
  genus_to_labels = dict(zip(list(class_names_dict_geni.keys()), range(len(class_names_dict_geni))))

  class_names_dict_familys = dict.fromkeys(file_dict_familys.values())
  family_to_labels = dict(zip(list(class_names_dict_familys.keys()), range(len(class_names_dict_familys))))

  class_names_dict_contents = dict.fromkeys(file_dict_contents.values())
  content_to_labels = dict(zip(list(class_names_dict_contents.keys()), range(len(class_names_dict_contents))))

  class_id_to_species = dict(zip(list(file_dict_ids.values()), list(file_dict_species.values())))

  # Divide into train and test:
  photo_filenames = list(file_dict_familys.keys())
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)

  training_filenames = photo_filenames[_NUM_VALIDATION:]
  validation_filenames = photo_filenames[:_NUM_VALIDATION]




  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, file_dict_geni,file_dict_familys,file_dict_ids,file_dict_contents, class_id_to_labels, genus_to_labels, family_to_labels, content_to_labels, dataset_dir, dataset_name)
  _convert_dataset('validation', validation_filenames, file_dict_geni,file_dict_familys,file_dict_ids,file_dict_contents, class_id_to_labels, genus_to_labels, family_to_labels, content_to_labels, dataset_dir, dataset_name)
  # _convert_dataset('test', photo_filenames, file_dict_geni,file_dict_familys,file_dict_ids,file_dict_contents, class_id_to_labels, genus_to_labels, family_to_labels, content_to_labels, dataset_dir, dataset_name)


  # Finally, write the labels file:
  label_to_class_ids = {v:k for k, v in class_id_to_labels.items()}
  label_to_geni = {v:k for k, v in genus_to_labels.items()}
  label_to_familys = {v:k for k, v in family_to_labels.items()}
  label_to_contents = {v:k for k, v in content_to_labels.items()}

  dataset_utils.write_label_file(label_to_class_ids, dataset_dir, filename='labels.txt')
  dataset_utils.write_label_file(label_to_geni, dataset_dir, filename='label_to_genus.txt')
  dataset_utils.write_label_file(label_to_familys, dataset_dir, filename='label_to_family.txt')
  dataset_utils.write_label_file(label_to_contents, dataset_dir, filename='label_to_content.txt')
  dataset_utils.write_label_file(class_id_to_species, dataset_dir, filename='class_id_to_species.txt')

  _clean_up_temporary_files(dataset_dir,dataset_name)
  print('\nFinished converting the Plantclef dataset!')



