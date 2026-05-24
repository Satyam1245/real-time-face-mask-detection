from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


EXPECTED_CLASSES = ("with_mask", "without_mask")
IMAGE_SIZE = (128, 128)
AUTOTUNE = tf.data.AUTOTUNE


def count_images(folder: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not folder.exists():
        return 0
    return sum(1 for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def validate_dataset(dataset_dir: Path, min_images_per_class: int) -> dict[str, int]:
    counts = {class_name: count_images(dataset_dir / class_name) for class_name in EXPECTED_CLASSES}
    missing = [class_name for class_name, count in counts.items() if count < min_images_per_class]
    if missing:
        details = ", ".join(f"{name}={counts[name]}" for name in EXPECTED_CLASSES)
        raise ValueError(
            "Dataset is not ready for binary mask training. "
            f"Expected at least {min_images_per_class} images per class in "
            f"{EXPECTED_CLASSES}; found {details}."
        )
    return counts


def build_model(input_shape: tuple[int, int, int]) -> keras.Model:
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Rescaling(1.0 / 255),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.Conv2D(32, 3, activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(256, 3, activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.35),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="face_mask_cnn",
    )


def prepare_dataset(dataset: tf.data.Dataset, training: bool) -> tf.data.Dataset:
    if training:
        dataset = dataset.shuffle(1000)
    return dataset.prefetch(AUTOTUNE)


def train(args: argparse.Namespace) -> None:
    dataset_dir = args.dataset.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = validate_dataset(dataset_dir, args.min_images_per_class)
    print("Dataset counts:", counts)

    train_ds = keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=args.validation_split,
        subset="training",
        seed=args.seed,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        label_mode="binary",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=args.validation_split,
        subset="validation",
        seed=args.seed,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        label_mode="binary",
    )

    class_names = train_ds.class_names
    train_ds = prepare_dataset(train_ds, training=True)
    val_ds = prepare_dataset(val_ds, training=False)

    model = build_model((*IMAGE_SIZE, 3))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            output_dir / "mask_detector.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    model.save(output_dir / "mask_detector.keras")

    label_map = {"class_names": class_names, "threshold": args.threshold, "image_size": IMAGE_SIZE}
    (output_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    print(f"Saved model to {output_dir / 'mask_detector.keras'}")
    print(f"Saved labels to {output_dir / 'label_map.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN for face mask detection.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-images-per-class", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
