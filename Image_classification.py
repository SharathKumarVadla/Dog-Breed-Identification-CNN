#from tensorflow import keras
from keras.models import load_model
import pickle
import numpy as np
import tensorflow as tf

import streamlit as st

def teachable_machine_classification(img, weights_file, labels_file):
    model = load_model(weights_file)
    st.text("model has been loaded")
    with open(labels_file,'rb') as file:
        encoder = pickle.load(file)
    st.text("Labels have been loaded")    
    labels = encoder.classes_    
    image = np.array(img)/255
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [400, 400])
    st.text("Image has been prepared for model")
    prediction = model.predict(image[np.newaxis,:,:,:],verbose=0)
    st.text("Prediction has been made")
    return labels[np.argmax(prediction)]
