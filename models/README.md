# Models Directory

This directory stores trained deep learning models used in the Real-Time Face Mask Detection project.

---

# Purpose

The trained model is responsible for classifying whether a detected face is:

- Wearing a Mask
- Not Wearing a Mask

The model is generated after training the CNN using the dataset provided in the project.

---

# Expected Files

Example model files:

```text
mask_detector.keras
mask_detector.h5
```

---

# Generate the Model

Run the training script:

```bash
python src/train.py
```

After successful training, the model file will be saved inside this directory.

---

# Notes

- Large model files may not be uploaded to GitHub due to storage limitations.
- The model can be retrained anytime using the provided dataset.
- Keep only the latest trained model to reduce unnecessary storage usage.

---

# Recommended Format

Preferred model format:

```text
.keras
```

This format provides better compatibility with modern TensorFlow and Keras versions.

---

# Usage

The trained model is used during:

- Real-time webcam detection
- Image prediction

Scripts using the model:

```text
src/detect_mask_video.py
src/predict_image.py
```

---

# Important

Do not manually modify trained model files unless required for advanced experimentation or optimization.