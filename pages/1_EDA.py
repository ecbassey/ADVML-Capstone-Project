import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import random

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
#st.write("Upload a tea leaf image and the model will classify it.")
st.write("\n") #Line break
st.write("\n") #Line break


#count
data_dir = "dataset"

st.title("Dataset Class Distribution")

counts = []

for cls in os.listdir(data_dir):
    cls_path = os.path.join(data_dir, cls)

    if os.path.isdir(cls_path):
        counts.append([cls, len(os.listdir(cls_path))])

df_counts = pd.DataFrame(counts, columns=["Class", "Count"])

st.dataframe(df_counts)

#st.bar_chart(df_counts.set_index("Class"))



#-------------------------------------------------
# Visual display
# dataset_path = "dataset"

# st.title("Class Preview")

# folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

# fig = plt.figure(figsize=(12, 6))

# for i, folder in enumerate(folders):
#     folder_path = os.path.join(dataset_path, folder)

#     image_files = os.listdir(folder_path)

#     if len(image_files) > 0:
#         image_file = image_files[0]  # safer than [1]

#         img = Image.open(os.path.join(folder_path, image_file))

#         plt.subplot(2, len(folders), i + 1)
#         plt.imshow(img)
#         plt.title(folder)
#         plt.axis('off')

# plt.tight_layout()

# st.pyplot(fig)

#----------------
st.title("Random Samples from Dataset")

dataset_path = "dataset"

folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

for folder in folders:
    folder_path = os.path.join(dataset_path, folder)

    image_files = os.listdir(folder_path)

    if len(image_files) == 0:
        continue

    fig = plt.figure(figsize=(10, 3))

    sample_files = random.sample(
        image_files,
        min(4, len(image_files))
    )

    for i, file in enumerate(sample_files):
        img = Image.open(os.path.join(folder_path, file))

        plt.subplot(1, 4, i + 1)
        plt.imshow(img)
        plt.axis('off')

    plt.suptitle(folder)
    st.pyplot(fig)


#-------------------
# bad images

st.title("Dataset Image Quality Check")

dataset_path = "dataset"

total_images = 0
corrupted_images = 0
corrupted_files = []

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            total_images += 1

            try:
                img = Image.open(file_path)
                img.verify()  # checks if image is corrupted
            except Exception:
                corrupted_images += 1
                corrupted_files.append(file)

# Display results in Streamlit
st.subheader("Results")

st.write(f"**Total images:** {total_images}")
st.write(f"**Corrupted images:** {corrupted_images}")

# Optional: show corrupted files list
if corrupted_files:
    st.warning("Corrupted files detected:")
    st.write(corrupted_files)
else:
    st.success("No corrupted images found")



# -------- Check brightness
st.title("Dataset Brightness Analysis")

dataset_path = "dataset"

brightness = []

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                brightness.append(np.mean(img))

if brightness:
    avg_brightness = np.mean(brightness)

    st.metric("Average Brightness", f"{avg_brightness:.2f}")

    st.write(f"Total images analyzed: {len(brightness)}")
else:
    st.warning("No valid images found in dataset.")
    st.write("Over 150 is good")