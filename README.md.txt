-   ECG Disease Detection Using EfficientNetB0
-   Project Overview

This project implements an ECG (Electrocardiogram) Disease Detection System using deep learning.
The model leverages EfficientNetB0, a state-of-the-art convolutional neural network, to classify ECG images into four categories:

- Abnormal Heartbeat

- History of Myocardial Infarction (MI)

- Myocardial Infarction (MI)

- Normal

The goal is to automatically detect cardiac abnormalities from ECG images to assist in early diagnosis.

-   Dataset

The dataset is structured as follows:

ECG_DATA/
├── train/
│   ├── Abnormal Heartbeat/
│   ├── History of MI/
│   ├── Myocardial Infarction/
│   └── Normal/
└── test/
    ├── Abnormal Heartbeat/
    ├── History of MI/
    ├── Myocardial Infarction/
    └── Normal/


Note: Dataset is not included due to size constraints.

Download the dataset from: https://www.kaggle.com/datasets/evilspirit05/ecg-analysis

-   Project Workflow

-   Data Preprocessing

Resize images to 224x224.

Apply EfficientNetB0 preprocessing for normalization.

-    Model Architecture

Base: EfficientNetB0 pretrained on ImageNet (without top layers).

Added layers: Global Average Pooling → Dropout (0.3) → Dense layer with softmax activation (4 classes).

Fine-tuning: Last 100 layers of EfficientNetB0 unfrozen.

-   Training

ImageDataGenerator with 80/20 train-validation split.

Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.

Optimizer: Adam (lr=1e-4)

Loss: Categorical Crossentropy

Epochs: 20

-   Evaluation

Metrics: Accuracy, Confusion Matrix, Classification Report

-   Prediction

Use predict_ecg_image(img_path, model, class_indices) to classify a single ECG image.

-   Installation
git clone https://github.com/nadeem1-git/ECG_Disease_Detection.git
cd ECG_Disease_Detection
python -m venv venv  # optional
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
pip install -r requirements.txt

-   Usage

Open train_ecg.ipynb in Google Colab.

Connect Google Drive to access the dataset:

from google.colab import drive
drive.mount('/content/drive')


Run all cells sequentially to:

Preprocess data

Train and fine-tune the model

Evaluate on test set

Visualize training metrics and confusion matrix

Predict a single ECG image:

predict_ecg_image("path_to_image.jpg", model, test_gen.class_indices)


Replace "path_to_image.jpg" with your test ECG image path. The predicted disease will be displayed.

-   Results

Training & Validation Accuracy and Loss plots

Confusion Matrix visualized using Seaborn

Sample predictions for uploaded ECG images

-   Author

Nadeem Noman

GitHub: https://github.com/nadeem1-git

LinkedIn: www.linkedin.com/in/nadeem-noman

Email: nadeemnoman227@gmail.com