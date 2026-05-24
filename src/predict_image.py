from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def load_labels(label_file: Path) -> tuple[list[str], float, tuple[int, int]]:
    metadata = json.loads(label_file.read_text(encoding="utf-8"))
    return metadata["class_names"], float(metadata.get("threshold", 0.5)), tuple(metadata.get("image_size", [128, 128]))


def preprocess_image(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    return np.expand_dims(image.astype("float32"), axis=0)


def classify(probability: float, class_names: list[str], threshold: float) -> tuple[str, float]:
    class_index = 1 if probability >= threshold else 0
    confidence = probability if class_index == 1 else 1.0 - probability
    return class_names[class_index], confidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict mask status for one face image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/mask_detector.keras"))
    parser.add_argument("--labels", type=Path, default=Path("models/label_map.json"))
    args = parser.parse_args()

    class_names, threshold, image_size = load_labels(args.labels)
    model = tf.keras.models.load_model(args.model)
    batch = preprocess_image(args.image, image_size)
    probability = float(model.predict(batch, verbose=0)[0][0])
    label, confidence = classify(probability, class_names, threshold)
    print(f"{label}: {confidence:.2%}")


if __name__ == "__main__":
    main()
