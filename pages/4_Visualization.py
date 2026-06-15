import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
import random


st.markdown("""
<style>
.stApp {
    background-color: #F1F8E9;
}
</style>
""", unsafe_allow_html=True)

st.title("Tea Leaf Disease Classification")
st.write("Upload a tea leaf image and the model will classify it.")


st.title("Accuracy/Loss graphs")

results_path = "results"

files = os.listdir(results_path)

models = sorted(set(f.split("_")[0] for f in files))

for model in models:
    st.subheader(model.upper())

    cols = st.columns(2)

    acc_file = f"{model}_accuracy.png"
    loss_file = f"{model}_loss.png"

    if os.path.exists(os.path.join(results_path, acc_file)):
        with cols[0]:
            st.image(Image.open(os.path.join(results_path, acc_file)),
                     caption="Accuracy",
                     use_container_width=True)

    if os.path.exists(os.path.join(results_path, loss_file)):
        with cols[1]:
            st.image(Image.open(os.path.join(results_path, loss_file)),
                     caption="Loss",
                     use_container_width=True)



# results_path = "results"

# accuracy_tabs, loss_tabs = st.tabs(["Accuracy", "Loss"])

# acc_files = [f for f in os.listdir(results_path) if "accuracy" in f]
# loss_files = [f for f in os.listdir(results_path) if "loss" in f]

# with accuracy_tabs:
#     for f in acc_files:
#         st.image(Image.open(os.path.join(results_path, f)), caption=f, use_container_width=True)

# with loss_tabs:
#     for f in loss_files:
#         st.image(Image.open(os.path.join(results_path, f)), caption=f, use_container_width=True)