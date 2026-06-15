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

model = tf.keras.models.load_model(
    "tea_leaf_model.keras"
)

# Make Prediction --------------
class_names = [
    "algal leaf",
    "brown blight",
    "healthy",
    "red leaf spot",
    "white spot"
]

# pred function
def predict_image(image):
    image = image.resize((224,224))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence

#Display Prediction
if uploaded_file is not None:
    pred, conf = predict_image(image)
    st.success(
        f"Prediction: {class_names[pred]}"
    )
    st.info(
        f"Confidence: {conf:.2%}"
    )


# #Show Model Performance  ---------------
# st.header("Model Comparison")

results = pd.DataFrame({
    "Model":[
        "CNN",
        "MobileNetV2",
        "VGG16",
        "ResNet50",
        "EfficientNetB0"
    ],
    "Accuracy":[
        0.79,
        0.37,
        0.44,
        0.22,
        0.20
    ]
})

st.dataframe(results)



# st.bar_chart(
#     results.set_index("Model")
# )


# st.image(
#     "results/model_comparison.png"
# )