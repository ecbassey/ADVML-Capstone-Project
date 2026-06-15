import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from PIL import Image
import os

st.markdown("""
<style>
.stApp {
    background-color: #F1F8E9;
}
</style>
""", unsafe_allow_html=True)

st.title("Tea Leaf Disease Classification")
st.write("Upload a tea leaf image and the model will classify it.")


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
        0.80,
        0.37,
        0.44,
        0.22,
        0.20
    ]
})

st.title("Model Evaluation")
st.subheader("Metrics Table")
st.dataframe(results)

#Bar Chart (best visual comparison)

#st.subheader("Accuracy Comparison")

# st.bar_chart(
#     results.set_index("Model")["Accuracy"]
# )

# -----------Classification report -------------------

models = {
    "CNN": tf.keras.models.load_model("tea_leaf_model.keras"),
    "MobileNetV2": tf.keras.models.load_model("mobilenet.keras"),
    "ResNet50": tf.keras.models.load_model("resnet.keras"),
}

# Function to generate report

#def evaluate_model(model, data, model_name, class_names):

# def get_report(model, data, class_names):
#     y_true = []
#     y_pred = []

#     for images, labels in data:
#         preds = model.predict(images)
#         y_pred.extend(np.argmax(preds, axis=1))
#         y_true.extend(labels.numpy())

#     report = classification_report(
#         y_true,
#         y_pred,
#         target_names=class_names,
#         output_dict=True
#     )

#     return pd.DataFrame(report).transpose()


# # show
# st.title("Model Classification Reports")

# model_choice = st.selectbox("Select Model", list(models.keys()))
# model = models[model_choice]


# st.subheader(f"{model_choice} Classification Report")

# report_df = get_report(model, data, class_names)

# st.dataframe(report_df)


# # all in one
# for name, model in models.items():
#     st.subheader(name)

#     report_df = get_report(model, data, class_names)

#     st.dataframe(report_df)


st.title("Classification Report")

st.subheader("Custom CNN")

# Accuracy display
st.metric(label="Accuracy", value="0.8042")

# Classification report
report_text = """
               precision    recall  f1-score   support

algal leaf       0.92      0.79      0.85        29
brown blight      0.67      0.53      0.59        30
healthy           0.97      1.00      0.98        29
red leaf spot     0.97      1.00      0.98        28
white spot        0.54      0.70      0.61        27

accuracy                              0.80       143
macro avg         0.81      0.81      0.80       143
weighted avg      0.81      0.80      0.80       143
"""

st.code(report_text)


#---------

st.subheader("ResNet50")

# Accuracy display
st.metric(label="Accuracy", value="0.2238")

# Classification report text
report_text = """
               precision    recall  f1-score   support

algal leaf       0.29      0.21      0.24        29
brown blight      0.00      0.00      0.00        30
healthy           0.19      0.52      0.28        29
red leaf spot     0.25      0.39      0.31        28
white spot        0.00      0.00      0.00        27

accuracy                              0.22       143
macro avg         0.15      0.22      0.17       143
weighted avg      0.15      0.22      0.17       143
"""

st.code(report_text)


# -----------vgg

st.subheader("VGG16")

# Accuracy display
st.metric(label="Accuracy", value="0.4476")

# Classification report text
report_text = """
               precision    recall  f1-score   support

algal leaf       0.35      0.76      0.48        29
brown blight      0.40      0.13      0.20        30
healthy           0.50      0.31      0.38        29
red leaf spot     0.59      0.71      0.65        28
white spot        0.47      0.33      0.39        27

accuracy                              0.45       143
macro avg         0.46      0.45      0.42       143
weighted avg      0.46      0.45      0.42       143
"""

st.code(report_text)


# ------------- Mobile

st.subheader("MobileNetV2")

# Accuracy display
st.metric(label="Accuracy", value="0.3776")

# Classification report text
report_text = """
               precision    recall  f1-score   support

algal leaf       0.30      0.62      0.40        29
brown blight      0.00      0.00      0.00        30
healthy           0.35      0.76      0.48        29
red leaf spot     0.67      0.50      0.57        28
white spot        0.00      0.00      0.00        27

accuracy                              0.38       143
macro avg         0.26      0.38      0.29       143
weighted avg      0.26      0.38      0.29       143
"""

st.code(report_text)





#-------------------



#---------------
st.title("Confusion Matrices")

results_path = "results"

cm_files = [f for f in os.listdir(results_path) if "Matrix" in f or "matrix" in f]

for file in cm_files:
    st.subheader(file.replace("_", " ").replace(".png", ""))

    img = Image.open(os.path.join(results_path, file))
    st.image(img, width=550)

