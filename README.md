# Real-Time Gender Classification using Convolutional Neural Networks (CNN) 🧑‍💻

This project implements a deep learning model to classify gender in real-time from a live video stream. It uses a Convolutional Neural Network (CNN) built with Python, TensorFlow, and OpenCV to accurately identify and label faces as 'Male' or 'Female' directly from a webcam feed.

## 🚀 Key Features

   Real-Time Detection & Classification: Processes live video from a webcam to detect faces and classify their gender simultaneously.
   Deep Learning Model: Utilizes a custom-trained Convolutional Neural Network (CNN) for high accuracy in classification.
   Lightweight & Efficient: Optimized for fast performance to enable smooth real-time processing on a standard computer.
   Simple User Interface: The output is displayed in a clean window showing the live video with bounding boxes and gender labels on detected faces.

## 🛠️ Technology Stack

   Programming Language: Python
   Deep Learning Framework: TensorFlow / Keras
   Computer Vision Library: OpenCV
   Numerical Operations: NumPy

## 🤔 How It Works

The project follows a standard machine learning pipeline:

1.  Data Collection: The model was trained on a large dataset of face images, each labeled with the corresponding gender.
2.  Preprocessing: Faces are detected in each frame, cropped, and resized to a uniform dimension suitable for the CNN input. The pixel values are normalized.
3.  CNN Model Architecture: The core of this project is the CNN. The network learns to extract hierarchical features from the face images. It starts by identifying simple features like edges and textures in the initial layers, and then combines them to recognize more complex features like eyes, noses, and eventually, the distinguishing characteristics of gender in deeper layers.
4.  Real-Time Inference: For live classification, the application captures video frames from the webcam. For each frame, it uses OpenCV's Haar Cascade classifier to detect faces. Each detected face is then preprocessed and fed into the trained CNN model, which predicts the gender. The resulting label is then drawn on the video frame.

## ⚙️ Setup and Installation

To get this project running on your local machine, follow these steps.

1.  Clone the Repository

    ```bash
    git clone https://github.com/sovanghoshh/real-time-gender-classification.git
    cd real-time-gender-classification
    ```

2.  Install Dependencies
    It's recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

    (Ensure `requirements.txt` includes `tensorflow`, `opencv-python`, and `numpy`).

3.  Download the Pre-trained Model
    Place the pre-trained model file (e.g., `gender_detection_model.h5`) and the Haar Cascade file (`haarcascade_frontalface_default.xml`) in the project's root directory.

-----

## ▶️ Usage

To start the real-time gender classification, run the main Python script from your terminal:

```bash
python run_app.py
```

A window will open displaying your webcam feed. When a face is detected, a green bounding box will appear around it with the predicted gender ('Male' or 'Female') labeled at the top. Press the 'q' key to quit the application.
