from tensorflow import keras
from keras.models import load_model
import pickle
import numpy as np
import tensorflow as tf

def teachable_machine_classification(img, weights_file,labels_file):
    model = load_model(weights_file)
    with open(labels_file,'rb') as file:
        encoder = pickle.load(file)
    labels = encoder.classes_    
    image = np.array(img)/255
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [400, 400])
    prediction = model.predict(image[np.newaxis,:,:,:],verbose=0)
    return labels[np.argmax(prediction)]
