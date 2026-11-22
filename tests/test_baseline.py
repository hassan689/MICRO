import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from baselines.difference_detector import (
    box_iou,
    detect_defects,
    evaluate_predictions,
    log_baseline_to_mlflow,
    read_dvc_version,
)


class BaselineTests(unittest.TestCase):
    def test_box_iou(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(box_iou([0, 0, 10, 10], [10, 10, 20, 20]), 0.0)

    def test_detects_difference_component(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            reference = np.full((40, 40, 3), 200, dtype=np.uint8)
            defective = reference.copy()
            defective[10:20, 10:20] = 50
            reference_path = directory / "sample_temp.jpg"
            defective_path = directory / "sample_test.jpg"
            Image.fromarray(reference).save(reference_path)
            Image.fromarray(defective).save(defective_path)

            result = detect_defects(
                defective_path,
                reference_path,
                threshold=30,
                min_area=20,
                kernel_size=3,
                blur_size=5,
            )
            self.assertEqual(len(result["boxes"]), 1)
            box = result["boxes"][0]
            self.assertGreaterEqual(box[0], 8)
            self.assertLessEqual(box[2], 21)

    def test_class_agnostic_matching(self):
        predictions = [{
            "boxes": [[0, 0, 10, 10], [20, 20, 30, 30]],
            "scores": [0.9, 0.8],
        }]
        targets = [[[0, 0, 10, 10], [20, 20, 30, 30]]]
        metrics = evaluate_predictions(predictions, targets)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_dvc_version_and_mlflow_logging(self):
        info = read_dvc_version()
        self.assertNotEqual(info["dvc_md5"], "unknown")
        self.assertNotEqual(info["dvc_rev_lock"], "unknown")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            directory = Path(directory)
            out_dir = directory / "outputs"
            out_dir.mkdir()
            (out_dir / "metrics.json").write_text("{}", encoding="utf-8")
            db_uri = f"sqlite:///{(directory / 'test_mlflow.db').as_posix()}"
            config = {
                "baseline": "aligned_grayscale_absdiff",
                "threshold": 30,
                "min_area": 20,
                "kernel_size": 3,
                "blur_size": 5,
                "iou_threshold": 0.5,
                "dataset": "DeepPCB",
                "source_sha256": "abc123",
                "split_sha256": "def456",
                "seed": 42,
                "val_fraction": 0.2,
                "validation_groups": ["group00041"],
                "train_indices": 800,
                "val_indices": 200,
            }
            metrics = {"precision": 0.8, "recall": 0.75, "f1": 0.774, "tp": 15, "fp": 4, "fn": 5}
            run_id = log_baseline_to_mlflow(config, metrics, out_dir, db_uri, "Test-Exp", "test-baseline")
            self.assertTrue(len(run_id) > 0)


if __name__ == "__main__":
    unittest.main()
