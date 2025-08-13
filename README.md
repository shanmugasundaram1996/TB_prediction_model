Tuberculosis Detection from Chest X-rays

This project presents a deep learning-based solution for **automated Tuberculosis (TB) diagnosis** using chest X-ray images. The goal is to assist medical professionals by providing a fast, reliable, and accessible diagnostic tool, especially in resource-constrained settings.

Dataset Sources
The model was trained using two publicly available datasets from Kaggle:
   TB Chest X-ray Dataset**: Contains labeled images of patients diagnosed with tuberculosis.
   Normal Chest X-ray Dataset**: Contains X-rays of healthy individuals.

Note: The normal dataset was significantly smaller than the TB dataset, which posed a challenge for model training.

Data Preprocessing & Augmentation
To address the **class imbalance**, extensive **image augmentation** techniques were applied to the normal X-ray images. These included:
- Rotation
- Flipping
- Zooming
- Brightness adjustment

This helped create a more balanced dataset and improved the model’s generalization capability.

Model Architecture & Selection
Several pre-trained convolutional neural networks (CNNs) were evaluated:
   VGG16**
   ResNet50**
   EfficientNetB0**

After rigorous testing and validation, VGG16 was selected as the final model due to its superior accuracy and consistency across test sets. Given the **medical nature** of the application, **accuracy and reliability** were prioritized over model complexity or inference speed.

Deployment
The final model was deployed as a web-based application using:
Streamlit: For building an interactive and user-friendly interface
AWS EC2: For hosting the application
Ubuntu OS: Chosen for its stability and compatibility with deployment tools

Users can upload chest X-ray images and receive instant predictions on whether the image indicates TB or not.


