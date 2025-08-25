# 🛒 Shoplifting Video Detection from Surveillance Videos

**This project focuses on detecting shoplifting activities from CCTV surveillance videos using advanced Deep Learning architectures.**

**Multiple models were implemented and compared, including 3D CNN, CNN-LSTM, and MAE Transformers. After thorough evaluation, the Pretrained CNN-LSTM model achieved the best accuracy and was deployed using Django for real-world usage.**

## 📂 Project Structure

📁 Shoplifting-Video-Detection/

├── dataset/ # Training & validation datasets

├── models/ # Model architectures & training scripts

│ ├── 3d_cnn.py

│ ├── cnn_lstm.py

│ ├── cnn_lstm_pretrained.py

│ └── mae_transformer.py

├── experiments/ # Training logs, checkpoints, and evaluation results

├── classifier/ # Django app for deployment

│ ├── inference.py

│ ├── views.py

│ ├── urls.py

│ └── templates/

│ └── index.html

├── static/ # CSS, JS, and frontend UI assets

├── media/ # Uploaded videos for testing

├── requirements.txt # Dependencies

├── manage.py # Django project entry point

├── README.md # Project documentation

└── report/ # Results & evaluation reports

## 📓 Model Overview

**The project explored multiple approaches for video classification**:

- 3D CNN (from scratch)

  - Learns spatial + temporal features directly from video frames.

- CNN-LSTM (from scratch)

  - Combines CNN for spatial features + LSTM for sequence modeling.

- CNN-LSTM (Pretrained CNN backbone) ✅ Best Model

  - Uses pretrained CNN (e.g., ResNet/VGG) for strong spatial feature extraction.

  - LSTM captures motion and temporal dependencies.

  - Achieved the highest accuracy:

  - Test Accuracy: 91 %


- MAE Transformers

  - Applied transformer-based video modeling.

  - Explored masked autoencoding for better temporal understanding.

## 🧪 Evaluation Metrics

- **We compared the performance of each model using accuracy and loss curves**.

Model	Accuracy
3D CNN (Scratch)	61 %
CNN-LSTM (Scratch)	50 %
CNN-LSTM (Pretrained)	91 %
MAE Transformers	49 %

### ✅ Best Model: CNN-LSTM (Pretrained)

## 🚀 Deployment

- **The best model (CNN-LSTM Pretrained) was deployed using a *Django* web application that supports:**

  - Uploading video clips.

  - Running inference in real-time.

  - Displaying results: Shoplifting or Normal.

## 🛠️ Tech Stack

  - Python 🐍

  - TensorFlow / Keras

  - PyTorch (for Transformer experiments)

  - OpenCV (video processing)

  - NumPy, Pandas

  - Django (for deployment)

## 🎯 Future Work

  - Integrate real-time CCTV camera streaming.

  - Improve transformer-based architectures for higher accuracy.

  - Optimize the model for edge devices (Raspberry Pi, Jetson Nano).

## ✨ Author

  - **Developed by *Ayman Abodala* and *Maryam Elsayed* — AI & Machine Learning Enthusiast.**
