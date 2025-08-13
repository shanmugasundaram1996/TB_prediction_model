import os
from PIL import Image
import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
import streamlit as st
import os
from PIL import Image
import tensorflow as tf
import numpy as np
import streamlit as st
import boto3


MODEL_PATH = "vgg16_tb_model.h5"
BUCKET_NAME = "xray-prediction-2025"
MODEL_KEY = "vgg16_tb_model.h5"

os.environ["AWS_ACCESS_KEY_ID"] ="AKIAQ2K25GY4GNRRW7PB"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

if not os.path.exists(MODEL_PATH):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.download_file(BUCKET_NAME, MODEL_KEY, MODEL_PATH)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

loaded_keras_model = load_model()


SAVE_DIR = "uploaded_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# @st.cache_resource
# def load_model():
#     model = tf.keras.models.load_model(r"C:\Users\shanm\OneDrive\Desktop\project\tbprediction\vgg16_tb_model.h5")
#     return model

# loaded_keras_model = load_model()




#set home page with titles of other pages
st.sidebar.title('HOME')
page=st.sidebar.radio("Getpage",["Project Info","View X-ray image","Final TB prediction  model ",
                                 "Diagnose The X-ray image",
                                    "Who Creates"])

if page=="Project Info":
    st.title("TB PREDICTION USING DEEP LEARNING")
    #put one image 
    st.image("C:/Users/shanm/OneDrive/Desktop/project/tbprediction/tb_detection_using_DL.jpg")
    # give a intro to the website
    st.write("""Tuberculosis (TB) remains one of the leading causes of mortality worldwide, 
              particularly in low-resource settings where timely diagnosis is critical yet often inaccessible. 
              Chest X-ray imaging is a widely used diagnostic tool for TB detection,
              but manual interpretation is time-consuming and prone to variability across radiologists.
              To address these challenges, deep learning offers a promising solution
              by automating and enhancing the accuracy of TB diagnosis.
 """)

    st.write("""This project leverages Convolutional Neural Networks (CNNs) — a class of deep learning models 
             highly effective in image analysis — to detect TB from chest X-ray images. 
             CNNs can learn hierarchical features directly from raw pixel data, 
             enabling robust pattern recognition in medical imaging.
 """)
    st.write("""To further improve performance and reduce training time, we employ transfer learning, 
             utilizing pretrained models such as ResNet50, VGG16, and EfficientNet. 
             These architectures, trained on large-scale datasets like ImageNet, 
             provide rich feature representations that can be fine-tuned for TB classification. 
             This approach is especially beneficial when working with limited labeled medical data.
 """)
elif page=="View X-ray image":
    st.header('Normal and TB X-ray ')
    st.image(r"C:\Users\shanm\OneDrive\Desktop\XRAY.png")   
    st.header('Normal X-ray from Dataset')
    st.image("C:/Users/shanm/OneDrive/Desktop/project/tbprediction/preprocessed_dataset/test/NORMAL/others (14).jpg")
    st.header('TB Positive X-ray from Dataset')
    st.image("C:/Users/shanm/OneDrive/Desktop/project/tbprediction/preprocessed_dataset/test/TB/TB.16.jpg")    

elif page=="Final TB prediction  model ":
    st.header('VGG16 Model for TB Prediction')
    st.image("C:/Users/shanm/OneDrive/Desktop/project/tbprediction/VGG_ARCHITECTURE.jpg")
    Query=st.selectbox("select below queries",["Confusion Matrix of VGG16 Model?","Accuracy of VGG16 Model?",
                                         "Loss of VGG16 Model?" 
                                         ])
    if Query=="Confusion Matrix of VGG16 Model?":
        st.image(r"C:\Users\shanm\OneDrive\Desktop\project\tbprediction\VGG_confusion_matrix.png")
    if Query=="Accuracy of VGG16 Model?":
        st.image(r"C:\Users\shanm\OneDrive\Desktop\project\tbprediction\model_accuracy_plot.png")
    if Query=="Loss of VGG16 Model?":
        st.image(r"C:\Users\shanm\OneDrive\Desktop\project\tbprediction\model_loss_plot.png")


elif page=="Diagnose The X-ray image":
    st.subheader("Give Details ")
  


# Create a directory to save uploaded images
    SAVE_DIR = "uploaded_images"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Upload image
    uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open image using PIL
        image = Image.open(uploaded_file)

        # Display image
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Save image to local directory
        image_path = os.path.join(SAVE_DIR, uploaded_file.name)
        image.save(image_path)



        
        IMG_SIZE = (224, 224)

        img = load_img(image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0

        # Ensure the image array has the correct shape for the VGG16 model (batch, height, width, channels)
        Z = np.expand_dims(img_array, axis=0) # Add batch dimension

        # Predict using the loaded model
        prediction_probabilities = loaded_keras_model.predict(Z)

        # Convert probabilities to binary classification using a threshold (e.g., 0.5)
        threshold = 0.5
        binary_predictions = (prediction_probabilities > threshold).astype(int)

        if binary_predictions==1:

           st.subheader("tb-positive")
        else:
           st.subheader("tb-negative ")

 


elif page=="Who Creates":
    col1, col2 = st.columns(2)			 																				

    with col1:
        st.image("https://thumbs.dreamstime.com/b/chibi-style-d-render-devops-engineer-playful-young-male-character-laptop-table-isolated-white-background-rendered-361524322.jpg")
    st.write("I am Shanmugasundaram, and this is my 2nd Machine Learning project after " \
        "joining the Data Science course on the Guvi platform. This project marks the beginning" \
        " of my journey into the world of data-driven decision-making and predictive analytics. " \
        "Through this project, I aim to apply the concepts learned in my course and build a model that provides " \
        "meaningful insights.")
    st.write("""Coming from an engineering background, I have always been intrigued by problem-solving,
                  automation, and analytical thinking. Machine Learning fascinates me as it combines mathematics, 
                 programming, and real-world applications to transform raw data into meaningful insights.""")
    













