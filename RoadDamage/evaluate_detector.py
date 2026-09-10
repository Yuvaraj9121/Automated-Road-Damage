"""Evaluate a trained Ultralytics detector on the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET_YAML = ROOT / "yolo_dataset" / "data.yaml"
ARTIFACTS = ROOT / "artifacts"


def as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate(model_path: Path, image_size: int, device: str) -> Path:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not DATASET_YAML.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATASET_YAML}"
        )

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=image_size,
        device=device,
        project=str(ARTIFACTS),
        name="evaluation",
        exist_ok=True,
        plots=True,
        verbose=False,
    )

    summary: dict[str, object] = {
        "model": str(model_path),
        "dataset": str(DATASET_YAML),
        "split": "test",
        "task": model.task,
        "classes": model.names,
        "metrics": {
            key: as_float(value)
            for key, value in metrics.results_dict.items()
        },
    }

    if hasattr(metrics, "box"):
        available_classes = min(
            len(model.names),
            len(metrics.box.p),
            len(metrics.box.r),
            len(metrics.box.f1),
            len(metrics.box.ap50),
            len(metrics.box.ap),
        )

        summary["per_class"] = {
            model.names[index]: {
                "precision": as_float(metrics.box.p[index]),
                "recall": as_float(metrics.box.r[index]),
                "f1": as_float(metrics.box.f1[index]),
                "map50": as_float(metrics.box.ap50[index]),
                "map50_95": as_float(metrics.box.ap[index]),
            }
            for index in range(available_classes)
        }

    ARTIFACTS.mkdir(exist_ok=True)

    output_path = ARTIFACTS / "evaluation.json"
    output_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "model" / "best.pt",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    print(evaluate(args.model, args.imgsz, args.device))


if __name__ == "__main__":
    main()