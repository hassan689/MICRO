"""Inspect a trained checkpoint on its saved validation split, without training."""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch
from torchvision.ops import box_iou
from torchvision.transforms.functional import to_pil_image

from evaluation.metrics import detection_metrics
from models.detector import build_detector
from preprocessing.dataset import CLASS_NAMES, PROJECT_ROOT


def match_predictions(prediction, target, confidence=0.5):
    """Class-aware, confidence-ordered one-to-one matching at IoU >= 0.5."""
    keep = torch.where(prediction["scores"] >= confidence)[0]
    keep = keep[prediction["scores"][keep].argsort(descending=True)].tolist()
    matched, correct, incorrect = set(), [], []
    for index in keep:
        same_class = torch.where(target["labels"] == prediction["labels"][index])[0]
        hit = False
        if len(same_class):
            overlaps = box_iou(prediction["boxes"][index:index+1], target["boxes"][same_class])[0]
            overlap, position = overlaps.max(0)
            target_index = int(same_class[position])
            if float(overlap) >= 0.5 and target_index not in matched:
                matched.add(target_index)
                correct.append(index)
                hit = True
        if not hit:
            incorrect.append(index)
    return {"correct": correct, "incorrect": incorrect,
            "missed": [i for i in range(len(target["labels"])) if i not in matched]}


def summarize(predictions, targets, confidence):
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASS_NAMES.values()}
    details = []
    for prediction, target in zip(predictions, targets):
        result = match_predictions(prediction, target, confidence)
        for field, source, label in [("correct", prediction, "tp"),
                                      ("incorrect", prediction, "fp"), ("missed", target, "fn")]:
            for index in result[field]:
                counts[CLASS_NAMES[int(source["labels"][index])]][label] += 1
        details.append(result)
    for value in counts.values():
        value["precision"] = value["tp"] / max(1, value["tp"] + value["fp"])
        value["recall"] = value["tp"] / max(1, value["tp"] + value["fn"])
    total = {key: sum(value[key] for value in counts.values()) for key in ("tp", "fp", "fn")}
    total["precision"] = total["tp"] / max(1, total["tp"] + total["fp"])
    total["recall"] = total["tp"] / max(1, total["tp"] + total["fn"])
    return {"confidence": confidence, "overall": total, "per_class": counts}, details


def render_example(image, target, prediction, matches, destination, name, confidence):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax in axes:
        ax.imshow(to_pil_image(image))
        ax.axis("off")

    def draw(ax, box, text, color):
        x1, y1, x2, y2 = box
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=1.6))
        ax.text(x1, max(0, y1-4), text, color=color, fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1})

    for index, (box, label) in enumerate(zip(target["boxes"].tolist(), target["labels"].tolist())):
        missed = index in matches["missed"]
        draw(axes[0], box, CLASS_NAMES[label] + (" MISSED" if missed else ""),
             "#bc2020" if missed else "#137638")
    for field, color in [("correct", "#137638"), ("incorrect", "#b45309")]:
        for index in matches[field]:
            label = CLASS_NAMES[int(prediction["labels"][index])]
            draw(axes[1], prediction["boxes"][index].tolist(),
                 f"{label} {float(prediction['scores'][index]):.2f}", color)
    axes[0].set_title("Ground truth: red = missed; green = matched")
    axes[1].set_title(f"Predictions >= {confidence}: green = correct; orange = false detection")
    figure.suptitle(name, fontsize=12)
    figure.subplots_adjust(top=.85, bottom=.04, left=.02, right=.98, wspace=.1)
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def review_checkpoint(run_dir, confidence=0.5):
    run_dir = Path(run_dir).resolve()
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    output = run_dir / f"validation_review_{confidence:g}"
    output.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs/matplotlib-cache"))
    torch.set_num_threads(2)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config.get("smoke_test"):
        raise ValueError("Select a full training run, not a smoke test")
    manifest = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    # Use the saved preprocessing implementation, not a later edited notebook.
    spec = importlib.util.spec_from_file_location("training_dataset_snapshot", run_dir / "dataset_source.py")
    snapshot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(snapshot)
    dataset = snapshot.DeepPCBDataset(root=config["root"])
    actual_hash = hashlib.sha256((dataset.root / "trainval.txt").read_bytes()).hexdigest()
    if actual_hash != manifest["source_sha256"]:
        raise ValueError("The raw split file changed since training")
    actual_paths = [p.relative_to(dataset.root).as_posix() for p, _ in dataset.samples]
    if actual_paths != manifest["sample_paths"]:
        raise ValueError("Dataset sample order changed since training")
    if set(manifest["train_indices"]) & set(manifest["val_indices"]):
        raise ValueError("Training and validation indices overlap")
    model = build_detector(pretrained=False, image_size=config["image_size"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for number, index in enumerate(manifest["val_indices"], 1):
            image, target = dataset[index]
            predictions.append(model([image])[0])
            targets.append(target)
            if number % 25 == 0:
                print(f"Reviewed {number}/{len(manifest['val_indices'])} validation images", flush=True)
    metrics = detection_metrics(predictions, targets)
    summary, details = summarize(predictions, targets, confidence)
    thresholds = [summarize(predictions, targets, threshold)[0] for threshold in (0.3, 0.5, 0.7)]
    rows = []
    for position, (index, matches) in enumerate(zip(manifest["val_indices"], details)):
        short_missed = sum(int(targets[position]["labels"][i]) == 2 for i in matches["missed"])
        short_false = sum(int(predictions[position]["labels"][i]) == 2 for i in matches["incorrect"])
        rows.append({"position": position, "dataset_index": index,
                     "image": actual_paths[index], "correct": len(matches["correct"]),
                     "false_detections": len(matches["incorrect"]), "missed": len(matches["missed"]),
                     "short_missed": short_missed, "short_false_detections": short_false})
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    worst = sorted(rows, key=lambda row: (row["short_missed"] + row["short_false_detections"], row["missed"]), reverse=True)
    selected = list(dict.fromkeys([0, len(rows)//2, worst[0]["position"], worst[1]["position"]]))
    gallery = []
    for position in selected:
        row = rows[position]
        image, target = dataset[row["dataset_index"]]
        destination = output / f"example_{position:03d}.png"
        reason = "Fixed example" if position in (0, len(rows)//2) else "Selected for high short-defect error count"
        render_example(image, target, predictions[position], details[position], destination,
                       f"{Path(row['image']).name} | {reason}", confidence)
        gallery.append({"file": destination.name, "reason": reason, **row})
    report = {"checkpoint": str(run_dir / "best.pt"), "validation_images": len(targets),
              "iou": 0.5, "ap50": metrics, "operating_point": summary,
              "threshold_comparison": thresholds, "examples": gallery,
              "saved_validation_map50": checkpoint["metrics"]["map50"],
              "note": "Validation only. Threshold examples are not test-set performance; model weights were not changed."}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Validation AP50: {metrics['map50']:.4f}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Report: {output / 'summary.json'}", flush=True)
    return output / "summary.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()
    review_checkpoint(args.run_dir, args.confidence)
