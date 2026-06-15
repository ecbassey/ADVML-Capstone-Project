import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import cv2
import pandas as pd
#from data_loader import DataLoader
#from data_cleaner import DataCleaner
#from analyzer import Analyzer

st.markdown("""
<style>
.stApp {
    background-color: #F1F8E9;
}
</style>
""", unsafe_allow_html=True)
# st.markdown("""
#     <style>
#     .stApp {
#         background-color: #b3e0dc;
#         color: white;
#     }
#      .custom-text {
#         font-size: 20px;
#         color: #1f1f1f;
#         font-weight: 500;
# }          

#     h1, h2, h3 {
#         color: ##00C4B4;
#     }
#     </style>
# """, unsafe_allow_html=True)


st.title("Tea Leaf Disease Classification")
#st.write("Upload a tea leaf image and the model will classify it.")


# -----------------------------------
st.markdown("## Problem Statement")

st.write("""
Tea farming is an important agricultural activity, but plant diseases can significantly reduce crop quality and yield. Traditionally, disease identification is performed manually by agricultural experts, which is time-consuming, costly, and prone to human error.

""")

#st.image("tea_leaf.jpg", width=200)
#-----------------------------

col1, col2 = st.columns(2)

with col1:
    st.image("tea_leaf.jpg", caption="Healthy Leaf", width=350)

with col2:
    st.image("tea_leaf2.jpg", caption="Diseased Leaf", width=350)

st.markdown("""
### Tea Leaf Disease Classification

This application uses deep learning models to classify tea leaves into different categories:
- Healthy
- Algal Disease 
- Brown Blight Disease
- Red Leaf Disease
- White Spot Disease 

""")


#-------------------------------
st.markdown("##  Dataset")

st.write("""The dataset contains 1 class of healthy tea leaves and 4 classes of Diseased tea leaves. 
         Each of the classes contains about 150 images.
         """)



# col1, col2 = st.columns(2)

# with col1:
#     st.write("5 million people directly and indirectly in Kenya")

# with col2:
#     st.write("TEst")




#---------------------------------------------------------    

st.markdown("## Insights and Solutions")

st.write("""
This project aims to:
""")

st.markdown("""

- Compare different deep learning architectures.
- Evaluate model performance using standard metrics.
- Identify the best-performing model.
- Develop a prediction pipeline for classifying new tea leaf images.
""")

st.write("")  # creates a space / line break
st.write("")  # creates a space / line break

#--------------------
#--------------------------

st.set_page_config(layout="wide")

# Custom CSS
st.markdown("""
<style>
.box1 {
    background-color: #d4edda;
    padding: 20px;
    border-radius: 10px;
    color: black;
    text-align: center;
}

.box2 {
    background-color: #d4edda;
    padding: 20px;
    border-radius: 10px;
    color: black;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Create columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="box1">
        <h3>Employment and Livelihoods</h3>
        <p>
        5 million people directly and indirectly in Kenya.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="box2">
        <h3>Exports</h3>
        <p>
        652.8 million kilograms in 2025, a 9.8% increase over 2024.
        </p>
    </div>
    """, unsafe_allow_html=True)


st.write("")  # creates a space / line break
st.write("")  # creates a space / line break
