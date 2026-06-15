import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import cv2
import pandas as pd

st.markdown("""
<style>
.stApp {
    background-color: #F1F8E9;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(layout="wide")
#st.title("Airline Delay Intelligence System")

st.title("Tea Leaf Disease Classification")
st.write("Upload a tea leaf image and the model will classify it.")
st.write("\n") #Line break
st.write("\n") #Line break




# Upload Image ------------------
uploaded_file = st.file_uploader(
    "Upload Tea Leaf Image",
    type=["jpg", "jpeg", "png"]
)



from PIL import Image

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True,
        width=250
    )

# # --------------------------------
# # Load Train Model ---------------------------
# #--------------------------------------

cnn_model = tf.keras.models.load_model("tea_leaf_model.keras")
mobile_model = tf.keras.models.load_model("mobilenet.keras")
#best_model = tf.keras.models.load_model("best_model.keras")
vgg_model = tf.keras.models.load_model("vgg.keras")

models_dict = {
    "CNN": cnn_model,
    "MobileNetV2": mobile_model,
    #"Best_Model": best_model,
    "VGG16": vgg_model
}



#---
selected_model_name = st.selectbox(
    "Choose Model",
    list(models_dict.keys())
)

model = models_dict[selected_model_name]



#---
class_names = [
    "algal leaf",
    "brown blight",
    "healthy",
    "red leaf spot",
    "white spot"
]

def predict(image, model):

    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    return class_names[class_index], confidence

# upload

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    label, conf = predict(image, model)

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {conf:.2%}")


# all together

def compare_all_models(image):

    results = {}

    for name, model in models_dict.items():

        label, conf = predict(image, model)

        results[name] = (label, conf)

    return results



if uploaded_file is not None:

    results = compare_all_models(image)

    for model_name, (label, conf) in results.items():

        st.write(f"### {model_name}")
        st.write(f"Prediction: {label}")
        st.write(f"Confidence: {conf:.2%}")