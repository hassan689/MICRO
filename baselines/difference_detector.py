import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data/raw/DeepPCB/PCBData"
DEFAULT_SPLIT = PROJECT_ROOT / "data/splits/deeppcb.json"
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
DEFAULT_EXPERIMENT = "DeepPCB-Defect-Detection"


def read_dvc_version(dvc_file=PROJECT_ROOT / "data/raw/DeepPCB.dvc"):
    path = Path(dvc_file)
    if not path.is_file():
        return {"dvc_md5": "unknown", "dvc_rev_lock": "unknown"}
    info = {"dvc_md5": "unknown", "dvc_rev_lock": "unknown"}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("rev_lock:"):
            info["dvc_rev_lock"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("md5:") and info["dvc_md5"] == "unknown":
            info["dvc_md5"] = stripped.split(":", 1)[1].strip()
    return info


def log_baseline_to_mlflow(config, metrics, output_dir, tracking_uri, experiment_name, run_name=None):
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    dvc_info = read_dvc_version()
    with mlflow.start_run(run_name=run_name or "baseline-aligned-absdiff") as run:
        mlflow.set_tags({
            "model_type": "classical_baseline",
            "stage": "baseline",
            "dataset": config["dataset"],
            "dvc_rev_lock": dvc_info["dvc_rev_lock"],
            "dvc_md5": dvc_info["dvc_md5"],
        })
        mlflow.log_params({
            "model_family": config["baseline"],
            "threshold": config["threshold"],
            "min_area": config["min_area"],
            "kernel_size": config["kernel_size"],
            "blur_size": config["blur_size"],
            "iou_threshold": config["iou_threshold"],
            "dataset": config["dataset"],
            "source_sha256": config["source_sha256"],
            "split_sha256": config["split_sha256"],
            "dvc_rev_lock": dvc_info["dvc_rev_lock"],
            "dvc_md5": dvc_info["dvc_md5"],
            "seed": config["seed"],
            "val_fraction": config["val_fraction"],
            "validation_groups": ",".join(config["validation_groups"]),
            "train_samples": config["train_indices"],
            "val_samples": config["val_indices"],
        })
        mlflow.log_metrics({
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "tp": float(metrics["tp"]),
            "fp": float(metrics["fp"]),
            "fn": float(metrics["fn"]),
        })
        mlflow.log_artifacts(str(output_dir), artifact_path="baseline_outputs")
        return run.info.run_id


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_defects(defective_path, reference_path, threshold=30,
                   min_area=20, kernel_size=3, blur_size=5):
    if blur_size < 1 or blur_size % 2 == 0:
        raise ValueError("blur_size must be a positive odd integer")
    if kernel_size < 1:
        raise ValueError("kernel_size must be positive")
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")
    if min_area < 1:
        raise ValueError("min_area must be positive")

    with Image.open(defective_path) as defective_image:
        defective = cv2.cvtColor(np.asarray(defective_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    with Image.open(reference_path) as reference_image:
        reference = cv2.cvtColor(np.asarray(reference_image.convert("RGB")), cv2.COLOR_RGB2GRAY)

    if defective.shape != reference.shape:
        raise ValueError(
            f"Aligned image size mismatch: {defective_path.name}={defective.shape}, "
            f"{reference_path.name}={reference.shape}"
        )

    difference = cv2.absdiff(defective, reference)
    difference = cv2.GaussianBlur(difference, (blur_size, blur_size), 0)
    _, mask = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    scores = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        if area >= min_area:
            component_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(component_mask, [contour], -1, 255, thickness=-1)
            mean_difference = float(cv2.mean(difference, mask=component_mask)[0])
            boxes.append([float(x), float(y), float(x + width), float(y + height)])
            scores.append(mean_difference / 255.0)
    return {"boxes": boxes, "scores": scores, "mask": mask}


def read_annotations(path):
    boxes = []
    labels = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"{path}:{line_number}: expected five integers")
            try:
                x1, y1, x2, y2, class_id = map(int, values)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: expected five integers") from error
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            labels.append(class_id)
    return boxes, labels


def box_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_predictions(predictions, targets, iou_threshold=0.5):
    if len(predictions) != len(targets):
        raise ValueError("Each prediction must have a matching target")
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    per_image = []

    for predicted, target in zip(predictions, targets):
        predicted_order = sorted(
            zip(predicted["boxes"], predicted["scores"]),
            key=lambda item: item[1],
            reverse=True,
        )
        target_boxes = list(target)
        matched = set()
        image_tp = 0

        for predicted_box, _ in predicted_order:
            best_index = None
            best_iou = iou_threshold
            for target_index, target_box in enumerate(target_boxes):
                if target_index in matched:
                    continue
                iou = box_iou(predicted_box, target_box)
                if iou >= best_iou:
                    best_iou = iou
                    best_index = target_index
            if best_index is None:
                false_positives += 1
            else:
                matched.add(best_index)
                image_tp += 1
                true_positives += 1

        image_fn = len(target_boxes) - len(matched)
        false_negatives += image_fn
        per_image.append({"tp": image_tp, "fp": len(predicted["boxes"]) - image_tp,
                          "fn": image_fn})

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "iou_threshold": iou_threshold,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_image": per_image,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the classical aligned-image-difference baseline")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--blur-size", type=int, default=5)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "outputs/baseline/difference-detector")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI,
                        help="MLflow tracking URI (default: sqlite:///.../mlflow.db)")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT,
                        help="MLflow experiment name")
    parser.add_argument("--run-name", default="baseline-aligned-absdiff",
                        help="MLflow run name")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow run logging")
    args = parser.parse_args()
    if not 0 <= args.iou <= 1:
        parser.error("iou must be between 0 and 1")

    root = args.root.resolve()
    split_path = args.split_file.resolve()
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    source = root / "trainval.txt"
    if not source.is_file():
        raise FileNotFoundError(f"Missing split file: {source}")
    actual_hash = sha256(source)
    if actual_hash != manifest["source_sha256"]:
        raise ValueError("The raw split file changed since this split was created")
    if set(manifest["train_indices"]) & set(manifest["val_indices"]):
        raise ValueError("Training and validation indices overlap")

    predictions = []
    targets = []
    prediction_rows = []
    for val_position, dataset_index in enumerate(manifest["val_indices"]):
        relative_path = manifest["sample_paths"][dataset_index]
        base = Path(relative_path)
        if base.stem.endswith("_test"):
            annotation_stem = base.stem[:-5]
            defective_path = root / base
            reference_path = root / base.with_name(f"{annotation_stem}_temp.jpg")
        else:
            annotation_stem = base.stem
            defective_path = root / base.with_name(f"{base.stem}_test.jpg")
            reference_path = root / base.with_name(f"{base.stem}_temp.jpg")
        annotation_path = root / base.with_name(f"{annotation_stem}.txt")
        if not annotation_path.is_file():
            alt_parent = base.parent.with_name(f"{base.parent.name}_not")
            annotation_path = root / alt_parent / f"{annotation_stem}.txt"
        for path in (defective_path, reference_path, annotation_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing dataset file: {path}")

        target_boxes, labels = read_annotations(annotation_path)
        result = detect_defects(
            defective_path,
            reference_path,
            threshold=args.threshold,
            min_area=args.min_area,
            kernel_size=args.kernel_size,
            blur_size=args.blur_size,
        )
        predictions.append({"boxes": result["boxes"], "scores": result["scores"]})
        targets.append(target_boxes)
        prediction_rows.append({
            "dataset_index": dataset_index,
            "val_position": val_position,
            "image": relative_path,
            "predicted_boxes": result["boxes"],
            "target_boxes": target_boxes,
            "target_labels": labels,
        })

    metrics = evaluate_predictions(predictions, targets, args.iou)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "baseline": "aligned_grayscale_absdiff",
        "threshold": args.threshold,
        "min_area": args.min_area,
        "kernel_size": args.kernel_size,
        "blur_size": args.blur_size,
        "iou_threshold": args.iou,
        "dataset": "DeepPCB",
        "source_sha256": actual_hash,
        "split_file": str(split_path),
        "split_sha256": sha256(split_path),
        "seed": manifest["seed"],
        "val_fraction": manifest["val_fraction"],
        "validation_groups": manifest["val_groups"],
        "train_indices": len(manifest["train_indices"]),
        "val_indices": len(manifest["val_indices"]),
    }
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "predictions.json").write_text(
        json.dumps(prediction_rows, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset_index", "image", "tp", "fp", "fn"])
        writer.writeheader()
        for row, image_metrics in zip(prediction_rows, metrics["per_image"]):
            writer.writerow({
                "dataset_index": row["dataset_index"],
                "image": row["image"],
                **image_metrics,
            })

    print(json.dumps(metrics, indent=2))
    print(f"Baseline artifacts: {args.output}")
    if not args.no_mlflow:
        run_id = log_baseline_to_mlflow(
            config=config,
            metrics=metrics,
            output_dir=args.output,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
        )
        print(f"MLflow baseline run logged: {run_id} ({args.tracking_uri})")


if __name__ == "__main__":
    main()
