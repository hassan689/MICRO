"""Single-command reproducible pipeline runner:
1. Validate raw & processed dataset schema (preprocessing.prepare)
2. Verify/create leakage-free group-disjoint split (preprocessing.splits)
3. Run classical aligned-difference baseline & log to MLflow (baselines.difference_detector)
4. Train Faster R-CNN detector, log metrics to MLflow, and register best model (training.train)
5. Export MLflow run logs & registered model table (evaluation.export_mlflow)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(title, command):
    print(f"\n=== {title} ===", flush=True)
    print(f"$ {' '.join(str(part) for part in command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data/raw/DeepPCB/PCBData")
    parser.add_argument("--manifest-output", type=Path, default=PROJECT_ROOT / "data/processed/deeppcb_manifest.json")
    parser.add_argument("--split-file", type=Path, default=PROJECT_ROOT / "data/splits/deeppcb.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick end-to-end verification (1 epoch on 2 train and 2 val images)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Initialize detector without downloading pretrained weights")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    parser.add_argument("--experiment-name", default="DeepPCB-Defect-Detection")
    parser.add_argument("--register-model", default="DeepPCB-FasterRCNN-Detector")
    parser.add_argument("--training-output", type=Path, default=PROJECT_ROOT / "outputs/training/reproduced-run")
    args = parser.parse_args()

    py = sys.executable

    # Step 1: Automated Schema & Quality Validation
    run_step(
        "Step 1/5: Data Validation & Manifest Generation",
        [
            py, "-m", "preprocessing.prepare",
            "--raw-root", str(args.raw_root),
            "--output", str(args.manifest_output),
        ],
    )

    # Step 2: Leakage-Free Group-Disjoint Split Verification
    run_step(
        "Step 2/5: Leakage-Free Group-Disjoint Split Verification",
        [
            py, "-m", "preprocessing.splits",
            "--root", str(args.raw_root),
            "--output", str(args.split_file),
            "--seed", str(args.seed),
        ],
    )

    # Step 3: Classical Baseline Evaluation & MLflow Logging
    run_step(
        "Step 3/5: Classical Difference-Detector Baseline + MLflow Logging",
        [
            py, "-m", "baselines.difference_detector",
            "--root", str(args.raw_root),
            "--split-file", str(args.split_file),
            "--tracking-uri", args.tracking_uri,
            "--experiment-name", args.experiment_name,
        ],
    )

    # Step 4: Primary Faster R-CNN Training + MLflow Tracking & Model Registration
    if args.training_output.exists():
        shutil.rmtree(args.training_output)
    train_cmd = [
        py, "-m", "training.train",
        "--root", str(args.raw_root),
        "--split-file", str(args.split_file),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--image-size", str(args.image_size),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--device", args.device,
        "--output", str(args.training_output),
        "--tracking-uri", args.tracking_uri,
        "--experiment-name", args.experiment_name,
        "--register-model", args.register_model,
    ]
    if args.smoke_test:
        train_cmd.append("--smoke-test")
    if args.no_pretrained:
        train_cmd.append("--no-pretrained")
    run_step(
        "Step 4/5: Primary Faster R-CNN Training & MLflow Model Registration",
        train_cmd,
    )

    # Step 5: Export MLflow Runs & Model Registry Summary for Submission
    run_step(
        "Step 5/5: Export MLflow Runs & Model Registry Report",
        [
            py, "-m", "evaluation.export_mlflow",
            "--tracking-uri", args.tracking_uri,
            "--experiment-name", args.experiment_name,
            "--model-name", args.register_model,
        ],
    )

    print("\nPipeline completed successfully!", flush=True)


if __name__ == "__main__":
    main()
