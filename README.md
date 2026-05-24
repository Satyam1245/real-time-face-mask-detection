# Real-Time Face Mask Detection

A deep learning and computer vision project that detects whether a person is wearing a face mask in real time using webcam input.

The system uses TensorFlow, OpenCV, and Convolutional Neural Networks (CNNs) for face detection and mask classification.

---

# Features

- Real-time face mask detection
- Webcam-based live prediction
- CNN-powered image classification
- Lightweight and beginner-friendly implementation
- Image prediction support
- Organized project structure

---

# Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

# Project Structure

```text
real-time-face-mask-detection/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── models/
│
├── src/
│   ├── train.py
│   ├── detect_mask_video.py
│   ├── predict_image.py
│   └── data_audit.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── setup.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Satyam1245/real-time-face-mask-detection.git
```

Move into the project directory:

```bash
cd real-time-face-mask-detection
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset

The dataset contains two classes:

- With Mask
- Without Mask

The images are used to train the CNN model for mask classification.

---

# Train the Model

Run the training script:

```bash
python src/train.py
```

The trained model will be stored inside the `models/` directory.

---

# Run Real-Time Detection

Start webcam-based face mask detection:

```bash
python src/detect_mask_video.py
```

The system will:
- open webcam feed
- detect faces
- classify mask/no-mask
- display predictions in real time

---

# Predict From an Image

Run image prediction:

```bash
python src/predict_image.py
```

This script predicts whether the selected image contains a masked or unmasked face.

---

# Future Improvements

- Mobile deployment
- Better face detection models
- Multi-face tracking
- Improved real-time performance
- Cloud deployment support

---

# Learning Outcomes

This project demonstrates practical implementation of:

- Computer Vision
- Deep Learning
- CNN-based image classification
- OpenCV face detection
- Real-time video processing
- TensorFlow model training

---

# License

This project is licensed under the MIT License.

---

# Author

Developed for educational and practical learning purposes in machine learning and computer vision.