"""Check group separation, detection scoring, and the seven-class head."""

import unittest

import torch

from evaluation.metrics import detection_metrics
from models.detector import build_detector
from preprocessing.splits import grouped_indices


class SplitTests(unittest.TestCase):
    def test_groups_disjoint_and_all_samples_used(self):
        groups = ["a"] * 60 + ["b"] * 20 + ["c"] * 10 + ["d"] * 10
        train, val = grouped_indices(groups, val_fraction=0.2, seed=42)
        self.assertEqual(len(val), 20)
        self.assertEqual(sorted(train + val), list(range(100)))
        self.assertFalse({groups[i] for i in train} & {groups[i] for i in val})
        self.assertEqual((train, val), grouped_indices(groups, 0.2, 42))

    def test_nearest_whole_group_size(self):
        train, val = grouped_indices(["a"] * 9 + ["b"] * 3)
        self.assertEqual((len(train), len(val)), (9, 3))
        with self.assertRaises(ValueError):
            grouped_indices(["a"] * 3)


class MetricTests(unittest.TestCase):
    def target(self, boxes, labels):
        return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(labels, dtype=torch.int64)}

    def prediction(self, boxes, labels, scores):
        return {**self.target(boxes, labels), "scores": torch.tensor(scores)}

    def test_perfect_and_wrong_class(self):
        target = self.target([[0, 0, 10, 10]], [1])
        correct = self.prediction([[0, 0, 10, 10]], [1], [0.9])
        self.assertEqual(detection_metrics([correct], [target])["map50"], 1.0)
        wrong = self.prediction([[0, 0, 10, 10]], [2], [0.9])
        self.assertEqual(detection_metrics([wrong], [target])["map50"], 0.0)

    def test_duplicate_is_false_positive(self):
        target = self.target([[0, 0, 10, 10], [20, 20, 30, 30]], [1, 1])
        pred = self.prediction([[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]],
                               [1, 1, 1], [0.9, 0.8, 0.7])
        self.assertAlmostEqual(detection_metrics([pred], [target])["map50"], 5 / 6)

    def test_missed_detections_and_empty_images(self):
        empty = self.prediction([], [], [])
        target = self.target([[0, 0, 10, 10]], [1])
        self.assertEqual(detection_metrics([empty], [target])["map50"], 0.0)
        self.assertEqual(detection_metrics([empty], [self.target([], [])])["map50"], 0.0)


class DetectorTests(unittest.TestCase):
    def test_custom_classes_and_image_size_without_download(self):
        model = build_detector(pretrained=False, image_size=320)
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, 7)
        self.assertEqual(model.roi_heads.box_predictor.bbox_pred.out_features, 28)
        self.assertEqual(model.transform.min_size, (320,))


if __name__ == "__main__":
    unittest.main()
