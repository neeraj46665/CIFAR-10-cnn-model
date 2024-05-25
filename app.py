import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Load your trained model here (assuming 'model' is already loaded)
model = tf.keras.models.load_model('cifar10_model_85acc.h5')

# Define CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Function to preprocess and predict image
def predict_image(image_file):
    img = image.load_img(image_file, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    return predicted_class

# Streamlit app
st.title('CIFAR-10 Image Classification')
uploaded_file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write('')
    st.write('Classifying...')

    # Get predicted class
    predicted_class = predict_image(uploaded_file)

    st.write('Prediction:')
    st.write(predicted_class)
