from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED_CLASSES = ("with_mask", "without_mask")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def audit_dataset(dataset_dir: Path) -> dict[str, int]:
    return {class_name: count_images(dataset_dir / class_name) for class_name in EXPECTED_CLASSES}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the face mask dataset folder.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    args = parser.parse_args()

    counts = audit_dataset(args.dataset)
    print(f"Dataset: {args.dataset.resolve()}")
    for class_name, count in counts.items():
        print(f"{class_name}: {count} images")

    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Missing labeled data for: {joined}")


if __name__ == "__main__":
    main()
