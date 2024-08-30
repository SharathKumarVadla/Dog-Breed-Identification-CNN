import streamlit as st

from keras.models import load_model
import pickle
import numpy as np
import tensorflow as tf

st.title("Dog Breed Identification")
st.text("Upload a Dog Image to know the name of Breed")

#from Image_classification import teachable_machine_classification
def teachable_machine_classification(img, weights_file, labels_file):
    st.text("model has started loading")
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

from PIL import Image

uploaded_file = st.file_uploader("Choose a Dog Image ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Dog Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    predicted_label = teachable_machine_classification(image, 'Inception_resnet_stacked_weights.h5', 'encoder')
    st.write('Dog Breed: '+ predicted_label)
