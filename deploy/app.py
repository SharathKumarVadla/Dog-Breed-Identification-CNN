import streamlit as st
st.title("Dog Breed Identification")
st.text("Upload a Dog Image to know the name of Breed")

from Image_classification import teachable_machine_classification
from PIL import Image

uploaded_file = st.file_uploader("Choose a Dog Image ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Dog Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    predicted_label = teachable_machine_classification(image, 'Inception_resnet_stacked_weights.h5','encoder')
    st.write('Dog Breed: '+ predicted_label)
