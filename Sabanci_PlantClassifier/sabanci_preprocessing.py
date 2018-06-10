from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim


def random_center_cropping(image, lower, upper):
  """
	Args:
    image: 3-D Tensor containing single image in [0, 1]
    lower: Minimum fraction of image to be defined as center
    upper: Maximum fraction of image to be defined as center

  Returns:
    a central cropped image
  Raises:
  """
  central_fraction = tf.random_uniform([],minval=lower, maxval=upper, dtype=tf.float32, name="uniform_dist")
  with tf.Session() as sess:
    central_cropped_image = tf.image.central_crop(image, sess.run(central_fraction))

  return central_cropped_image

def random_patch_cropping(image, crop_size):
  """Crops a random square patches of the image.

  Args:
    image: 3-D Tensor containing single image in [0, 1]
    crop_size: size of the square
  Returns:
    the  cropped image.
  """

  image_rank = tf.rank(image)
  original_shape = tf.shape(image)
  cropped_shape = tf.stack([crop_size, crop_size, original_shape[2]])

  original_height = original_shape[0]
  original_width = original_shape[1]

  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_height, crop_size),
          tf.greater_equal(original_width, crop_size)),
      ['Crop size greater than the image size.'])


  # Create a random bounding box.
  max_offset_height = control_flow_ops.with_dependencies(
      asserts, tf.reshape(original_height - crop_size + 1, []))
  max_offset_width = control_flow_ops.with_dependencies(
      asserts, tf.reshape(original_width - crop_size + 1, []))
  offset_height = tf.random_uniform(
      [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform(
      [], maxval=max_offset_width, dtype=tf.int32)

  cropped_shape = tf.stack([crop_size, crop_size, original_shape[2]])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  image = control_flow_ops.with_dependencies(
      [crop_size_assert],
      tf.slice(image, offsets, cropped_shape))

  return tf.reshape(image, cropped_shape)


def k_patch_random_cropping(image, k, lower, upper):
  """Crops k random square patches of the image around center.

  Args:
    image: a image tensors
    k: number of patches to crop
    lower: Minimum fraction of image to be defined as center
    upper: Maximum fraction of image to be defined as center

  Returns:
    the image_list with cropped images.
  """

  image_rank = tf.rank(image)

  original_shape = tf.shape(image)

  original_height = original_shape[0]
  original_width = original_shape[1]

  center_cropped_image = random_center_cropping(image,lower,upper)

  center_cropped_shape = tf.shape(center_cropped_image)

  center_cropped_height = center_cropped_shape[0]
  center_cropped_width = center_cropped_shape[1]

  maxval=tf.minimum(center_cropped_height,center_cropped_width)

  minval = tf.cast(tf.cast(maxval,tf.float32) * 0.5, tf.int32)
  patches = []
  for i in range(k):
    crop_size = tf.random_uniform([], maxval=maxval, minval=minval ,dtype=tf.int32)

    crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_height, crop_size),
          tf.greater_equal(original_width, crop_size)),
      ['Crop size greater than the image size.'])

    random_patch = control_flow_ops.with_dependencies([crop_size_assert],random_patch_cropping(center_cropped_image, crop_size))
    patches.append(random_patch)
  return patches


def corner_center_cropping(image, crop_size):
  """Crops 5 square patches of the image. One patch for each corner and one fot the center.

  Args:
    image: a image tensors
    crop_size: a number specifying the size of the square

  Returns:
    the image_list with cropped images.

  Raises:
  """
  image_rank = tf.rank(image)

  original_shape = tf.shape(image)

  original_height = original_shape[0]
  original_width = original_shape[1]

  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_height, crop_size),
          tf.greater_equal(original_width, crop_size)),
      ['Crop size greater than the image size.'])

  upper_left = tf.image.crop_to_bounding_box(image, 0, 0, crop_size, crop_size)
  upper_right = tf.image.crop_to_bounding_box(image, 0, original_width - crop_size, crop_size, crop_size)
  lower_left = tf.image.crop_to_bounding_box(image, original_height - crop_size, 0, crop_size, crop_size)
  lower_right = tf.image.crop_to_bounding_box(image, original_height - crop_size, original_width - crop_size, crop_size, crop_size)

  with tf.Session() as sess:
      with slim.queues.QueueRunners(sess):
        center = tf.image.central_crop(image, crop_size/sess.run(original_height))
  corner_center_patch = [upper_left, upper_right, lower_left, lower_right, center]
  return corner_center_patch


def get_patches(image, k, lower, upper, r, is_evaluation):
  """ Get a patch from preprocessing

  Args:
    image: a image tensors
    k: number of patches to crop
    lower: Minimum fraction of image to be defined as center
    upper: Maximum fraction of image to be defined as center
    r: list of degrees to rotate the image
    is_evaluation: boolean wether the preprocessing is used for evaluation

  Returns:
    a random choosen image patch
  """


  patches = [image] # original image

  k_patches = k_patch_random_cropping(image, k, lower, upper) # k patches around the center
  patches.extend(k_patches)


  rotated_images = []
  for i in r:
      rotated_images.append(tf.contrib.image.rotate(image, i)) # rotated image for + degree
      rotated_images.append(tf.contrib.image.rotate(image, -i)) # rotated image for - degree

  for rotated_image in rotated_images:
    min_dim_rotated = tf.minimum(tf.shape(rotated_image)[0],tf.shape(rotated_image)[1]) # smaller image side
    max_center_crop = tf.image.resize_image_with_crop_or_pad(rotated_image, target_height = min_dim_rotated , target_width = min_dim_rotated) # max center cropped image
    patches.append(max_center_crop)




  if is_evaluation: # generate all image patches for score level averaging
      cropped_and_reflected_patches = []
      for image in patches:
          resized_image = tf.image.resize_images(image,[256,256]) # scale up
          normalized_image = tf.image.per_image_standardization(resized_image) # normalize
          corner_center_cropped_patches = corner_center_cropping(normalized_image, 224) # crop corner and center for each image
          cropped_and_reflected_patches.extend(corner_center_cropped_patches)
          #for crop in corner_center_cropped_patches:
              #flipped_crop = tf.image.flip_up_down(crop) # mirror image
              #cropped_and_reflected_patches.extend([crop,flipped_crop]) # add mirrored and orginial version

      return cropped_and_reflected_patches

  else: # randomly generate a subset and choose one
    index_patch = np.random.choice(len(patches)) # randomly choose a image for further preprocessing

    random_choice_patch = patches[index_patch]

    resized_image = tf.image.resize_images(random_choice_patch,[256,256]) # scale up

    normalized_image = tf.image.per_image_standardization(resized_image) # normalize

    corner_center_cropped_patches = corner_center_cropping(normalized_image, 224) # crop corner and center

    cropped_and_reflected_patches = []
    #for crop in corner_center_cropped_patches:
        #flipped_crop = tf.image.flip_up_down(crop)
        #cropped_and_reflected_patches.extend([crop,flipped_crop])
    cropped_and_reflected_patches.extend(corner_center_cropped_patches)

    index_choice = np.random.choice(len(cropped_and_reflected_patches)) # randomly choose a image

    return cropped_and_reflected_patches[index_choice]










