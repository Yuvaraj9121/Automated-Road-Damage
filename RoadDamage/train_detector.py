"""Prepare and train the repository's Ultralytics YOLO detector."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "yolo_dataset"
CLASS_NAMES = ["Block crack", "D00", "D10", "D20", "D40", "Repair"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_split(split: str) -> tuple[list[Path], list[str]]:
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split
    errors: list[str] = []

    if not image_dir.is_dir():
        return [], [f"Missing image directory: {image_dir}"]
    if not label_dir.is_dir():
        return [], [f"Missing label directory: {label_dir}"]

    valid_images: list[Path] = []
    for image_path in image_files(image_dir):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            errors.append(f"Missing label for {image_path.name}")
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            errors.append(f"Unreadable image {image_path.name}: {exc}")
            continue

        try:
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                values = line.split()
                if len(values) != 5:
                    raise ValueError("expected class_id and four normalized coordinates")
                class_id = int(values[0])
                coordinates = [float(value) for value in values[1:]]
                if not 0 <= class_id < len(CLASS_NAMES):
                    raise ValueError(f"class_id {class_id} is outside 0..{len(CLASS_NAMES) - 1}")
                if any(value < 0 or value > 1 for value in coordinates):
                    raise ValueError("coordinates must be normalized to [0, 1]")
        except Exception as exc:
            errors.append(f"Invalid label {label_path.name}:{line_number}: {exc}")
            continue

        valid_images.append(image_path)

    return valid_images, errors


def copy_split(images: Iterable[Path], split: str) -> None:
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        shutil.copy2(image_path, image_dir / image_path.name)
        source_label = image_path.parent.parent.parent / "labels" / "train" / f"{image_path.stem}.txt"
        shutil.copy2(source_label, label_dir / f"{image_path.stem}.txt")


def prepare_dataset(seed: int = 42) -> dict[str, int]:
    if not DATASET.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET}")

    train_images, train_errors = validate_split("train")
    val_images, val_errors = validate_split("val")
    errors = train_errors + val_errors
    if errors:
        raise RuntimeError("Dataset validation failed:\n" + "\n".join(errors[:20]))

    test_images: list[Path] = []
    test_image_dir = DATASET / "images" / "test"
    test_label_dir = DATASET / "labels" / "test"
    if test_image_dir.exists() or test_label_dir.exists():
        test_images, test_errors = validate_split("test")
        if test_errors:
            raise RuntimeError("Existing test split is invalid:\n" + "\n".join(test_errors[:20]))

    if not test_images:
        if len(train_images) < 3:
            raise RuntimeError("At least three valid training images are required to create a test split")
        shuffled = train_images[:]
        random.Random(seed).shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * 0.15))
        test_images = shuffled[:test_count]
        remaining_train = shuffled[test_count:]
        for image_path in test_images:
            (DATASET / "images" / "test").mkdir(parents=True, exist_ok=True)
            (DATASET / "labels" / "test").mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, DATASET / "images" / "test" / image_path.name)
            shutil.copy2(
                DATASET / "labels" / "train" / f"{image_path.stem}.txt",
                DATASET / "labels" / "test" / f"{image_path.stem}.txt",
            )
            image_path.unlink()
            (DATASET / "labels" / "train" / f"{image_path.stem}.txt").unlink()
        train_images = remaining_train

    yaml_path = DATASET / "data.yaml"
    yaml_path.write_text(
        f"path: {DATASET.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        + "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
        + "\n",
        encoding="utf-8",
    )

    summary = {"train": len(train_images), "val": len(val_images), "test": len(test_images)}
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "dataset_summary.json").write_text(
        json.dumps({"seed": seed, "classes": CLASS_NAMES, "splits": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def train(model_name: str, epochs: int, image_size: int, device: str) -> None:
    from ultralytics import YOLO

    model = YOLO(model_name)
    model.train(
        data=str(DATASET / "data.yaml"),
        epochs=epochs,
        imgsz=image_size,
        device=device,
        project=str(ROOT / "artifacts"),
        name="road_damage",
        exist_ok=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Validate and split data without training")
    parser.add_argument("--model", default=str(ROOT / "yolov8n.pt"), help="YOLO checkpoint or model name")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    summary = prepare_dataset()
    print(json.dumps(summary, indent=2))
    if not args.prepare_only:
        train(args.model, args.epochs, args.imgsz, args.device)


if __name__ == "__main__":
    main()
