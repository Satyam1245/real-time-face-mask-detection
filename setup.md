# Setup Guide

Follow the steps below to run the project on your system.

---

# 1. Clone the Repository

```bash
git clone https://github.com/Satyam1245/real-time-face-mask-detection.git
```

---

# 2. Move Into the Project Directory

```bash
cd real-time-face-mask-detection
```

---

# 3. Create a Virtual Environment (Recommended)

## Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

# 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 6. Train the Model

Run the training script:

```bash
python src/train.py
```

The trained model will be saved inside the `models/` directory.

---

# 7. Run Real-Time Face Mask Detection

Start webcam detection:

```bash
python src/detect_mask_video.py
```

Press:

```text
q
```

to close the webcam window.

---

# 8. Predict From an Image

Run:

```bash
python src/predict_image.py
```

Ensure the image path is correctly configured inside the script.

---

# Requirements

- Python 3.9 or above
- Webcam for real-time detection
- Internet connection for dependency installation

---

# Common Issues

## Webcam Not Opening

Possible reasons:
- camera permissions disabled
- another application using the webcam

Close other camera applications and try again.

---

## TensorFlow Installation Error

Upgrade pip before installation:

```bash
python -m pip install --upgrade pip
```

Then reinstall dependencies.

---

## Module Not Found Error

Install missing packages manually:

```bash
pip install package_name
```

Example:

```bash
pip install opencv-python
```

---

# Recommended Environment

- VS Code
- Python 3.10
- Windows 10/11 or Linux

---

# Notes

- Keep dataset folders properly organized.
- Avoid changing file paths unless necessary.
- Use a virtual environment for dependency isolation.

---

# Support

This project is intended for educational and learning purposes in machine learning and computer vision.