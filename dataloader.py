import tensorflow as tf
import numpy as np

def tfdata_generator(images, labels, is_training, batch_size=16):
  def parse_function(filename,labels):
    #decoding JPG file
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = image/255
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [400,400])
    y = tf.one_hot(tf.cast(labels, tf.uint8), 120)
    return image,y

  def augment(image,labels):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=6, dtype=tf.int32))
    image = tf.image.random_hue(image, 0.5)
    image = tf.image.random_saturation(image, 0.5, 2.5)
    image = tf.image.random_brightness(image, 0.5)
    image = tf.image.random_contrast(image, 0.1, 2)
    
    return image,labels
  
  dataset = tf.data.Dataset.from_tensor_slices((images,labels))
  if is_training:
    dataset = dataset.shuffle(18000) # depends on sample size
  # Transform and batch data at the same time
  dataset = dataset.map(parse_function)
  dataset = dataset.batch(batch_size, drop_remainder=True if is_training else False)
  
  if is_training:
    if np.random.uniform(0,1)>0.6:
      dataset = dataset.map(augment,num_parallel_calls=4)
  dataset = dataset.repeat()
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset