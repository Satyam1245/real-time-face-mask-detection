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


def classify_face(model: tf.keras.Model, face_bgr: np.ndarray, class_names: list[str], threshold: float, image_size: tuple[int, int]) -> tuple[str, float]:
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, image_size)
    batch = np.expand_dims(face_rgb.astype("float32"), axis=0)
    probability = float(model.predict(batch, verbose=0)[0][0])
    class_index = 1 if probability >= threshold else 0
    confidence = probability if class_index == 1 else 1.0 - probability
    return class_names[class_index], confidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-time face mask detection from webcam video.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=Path("models/mask_detector.keras"))
    parser.add_argument("--labels", type=Path, default=Path("models/label_map.json"))
    parser.add_argument("--scale-factor", type=float, default=1.1)
    parser.add_argument("--min-neighbors", type=int, default=5)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}. Train first with python src/train.py")
    if not args.labels.exists():
        raise SystemExit(f"Label map not found: {args.labels}. Train first with python src/train.py")

    class_names, threshold, image_size = load_labels(args.labels)
    model = tf.keras.models.load_model(args.model)

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(str(cascade_path))
    if face_detector.empty():
        raise SystemExit(f"Could not load Haar cascade: {cascade_path}")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=args.scale_factor,
                minNeighbors=args.min_neighbors,
                minSize=(60, 60),
            )

            for x, y, width, height in faces:
                face = frame[y : y + height, x : x + width]
                label, confidence = classify_face(model, face, class_names, threshold, image_size)
                color = (0, 180, 0) if label == "with_mask" else (0, 0, 220)
                text = f"{label} {confidence:.0%}"
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(frame, text, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Real-Time Face Mask Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
