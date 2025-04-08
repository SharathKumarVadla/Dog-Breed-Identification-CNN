from tensorflow import keras
from keras.models import load_model
import pickle
import numpy as np
import tensorflow as tf
import gdown
import os

# Google Drive URL for your model
MODEL_URL = 'https://drive.google.com/uc?id=1B9Az0jwM13QBdAoFJOQkQL4vwPFyHjlZ'

def download_file_if_missing(url, filepath):
    if not os.path.exists(filepath):
        gdown.download(url, filepath, quiet=False)

def teachable_machine_classification(img, weights_file,labels_file):
    # Ensure model is downloaded
    download_file_if_missing(MODEL_URL, weights_file)
    
    model = load_model(weights_file)
    with open(labels_file,'rb') as file:
        encoder = pickle.load(file)
    labels = encoder.classes_    
    image = np.array(img)/255
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [400, 400])
    prediction = model.predict(image[np.newaxis,:,:,:],verbose=0)
    return labels[np.argmax(prediction)]
