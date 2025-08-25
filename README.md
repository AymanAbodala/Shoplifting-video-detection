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

myproject/                        # اسم المشروع الرئيسي

│

├── manage.py                     # سكريبت إدارة Django

│

├── myproject/                     # مجلد الإعدادات (settings.py وغيره)

│   ├── __init__.py

│   ├── asgi.py

│   ├── settings.py               # إعدادات المشروع

│   ├── urls.py                   # الروابط الرئيسية

│   ├── wsgi.py

│

├── app/                           # التطبيق اللي بيحتوي على الموديل

│   ├── __init__.py

│   ├── admin.py

│   ├── apps.py

│   ├── migrations/

│   ├── models.py                  # (اختياري لو عندك DB)

│   ├── views.py                   # هنا بتكتب الـ API اللي بيشغل الموديل

│   ├── urls.py                    # روابط التطبيق

│   ├── forms.py                   # (اختياري لو عندك فورم HTML)

│   ├── templates/                 # صفحات HTML

│   │   └── index.html             # صفحة الواجهة

│   ├── static/                    # ملفات CSS/JS

│   │   ├── css/

│   │   └── js/

│   └── ml_model/                  # مجلد خاص بالموديل

│       ├── __init__.py

│       ├── model.py               # كود لتحميل الموديل والتنبؤ

│       └── saved_model/           # ملفات الموديل (pth / h5 / pkl...)

│

├── requirements.txt               # مكتبات Python

└── runtime.txt / Procfile          # ملفات خاصة بـ Deployment (مثلاً Heroku)


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

- **Model	Accuracy**
  - 3D CNN (Scratch)	61 %
  - CNN-LSTM (Scratch)	50 %
  - CNN-LSTM (Pretrained)	91 %
  - MAE Transformers	49 %

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
