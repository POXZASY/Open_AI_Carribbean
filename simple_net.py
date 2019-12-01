import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import os

IMG_WIDTH = 128
IMG_HEIGHT = 128

#https://www.tensorflow.org/guide/eager
tf.enable_eager_execution()

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_bmp(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label
####################
#CREATE THE DATASET#
####################
#https://www.tensorflow.org/tutorials/load_data/images

data_dir = pathlib.Path('./100_test')
image_count = len(list(data_dir.glob('*/*.bmp')))
print(image_count)
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
print(CLASS_NAMES)

#list of file paths
list_ds = tf.data.Dataset.list_files(str(data_dir/'*.*'))

#dataset of image / label pairs
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
#-1 for tf.data.experimental.AUTOTUNE
labeled_ds = list_ds.map(process_path, num_parallel_calls=5)


