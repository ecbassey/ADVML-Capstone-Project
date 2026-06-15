import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

st.markdown("""
<style>
.stApp {
    background-color: #F1F8E9;
}
</style>
""", unsafe_allow_html=True)

st.title("Model Training Overview")

st.subheader("Model Architecture")

st.write("**Base model:** custom CNN / VGG16 / EfficientNetB3 / MobileNetV2 / ResNet50")
st.write("**Layers:** Conv2D, MaxPooling, Dense, Dropout")
st.write("**Activation:** ReLU (hidden layers), Softmax (output)")

st.subheader("Train/Validation Split")
st.write("**Split:** 80/20 using stratified sampling")


st.subheader("Data Preprocessing")
st.write("**Rescaling:** 1/255")
st.write("**Augmentation:** rotation_range / zoom_range / horizontal_flip / shear_range")


st.write("**Image Size:** 224 x 224")
st.write("**Batch Size:** 25")
st.write("**Loss Function:** Categorical Crossentropy")
st.write("**Epochs:** 15 - 20")
st.write("**Shuffle:** False")
st.write("**Patience:** 5")


st.subheader("Learning Rate")
st.write("**Optimizer:** Adam")
st.write("**Default Adam LR = 0.001 to 0.0005**")


st.subheader("Environment")
st.write("**TensorFlow**")



cnn_history = np.load("history.npy", allow_pickle=True).item()

st.subheader("CNN Final Results")

st.metric("Final Training Accuracy", f"{cnn_history['accuracy'][-1]:.2f}")
st.metric("Final Validation Accuracy", f"{cnn_history['val_accuracy'][-1]:.2f}")

with st.expander("Show Training Details"):
    st.write(cnn_history)

best_epoch = np.argmax(cnn_history["val_accuracy"])
st.write("Best Epoch:", best_epoch)

if cnn_history["val_accuracy"][-1] < 0.6:
    st.warning("Model may be underfitting")

# vgg

vgg_history = np.load("vgg_history.npy", allow_pickle=True).item()

st.subheader("VGG Final Results")

st.metric("Final Training Accuracy", f"{vgg_history['accuracy'][-1]:.2f}")
st.metric("Final Validation Accuracy", f"{vgg_history['val_accuracy'][-1]:.2f}")

with st.expander("Show Training Details"):
    st.write(vgg_history)

best_epoch = np.argmax(vgg_history["val_accuracy"])
st.write("Best Epoch:", best_epoch)

if vgg_history["val_accuracy"][-1] < 0.6:
    st.warning("Model may be underfitting")


# mobile

mobile_history = np.load("mobile_history.npy", allow_pickle=True).item()

st.subheader("Mobile Final Results")

st.metric("Final Training Accuracy", f"{mobile_history['accuracy'][-1]:.2f}")
st.metric("Final Validation Accuracy", f"{mobile_history['val_accuracy'][-1]:.2f}")

with st.expander("Show Training Details"):
    st.write(mobile_history)

best_epoch = np.argmax(mobile_history["val_accuracy"])
st.write("Best Epoch:", best_epoch)

if mobile_history["val_accuracy"][-1] < 0.6:
    st.warning("Model may be underfitting")




# st.subheader("Training vs Validation Accuracy")

# fig, ax = plt.subplots()

# ax.plot(cnn_history["accuracy"], label="Train Accuracy")
# ax.plot(cnn_history["val_accuracy"], label="Validation Accuracy")

# ax.set_xlabel("Epochs")
# ax.set_ylabel("Accuracy")
# ax.legend()

# st.pyplot(fig)



st.subheader("ACCURACY AND LOSS GRAPHS")