import unittest
import torch
from evaluation.review import match_predictions, summarize


class ReviewTests(unittest.TestCase):
    def test_duplicates_wrong_class_and_threshold(self):
        target = {"boxes": torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=torch.float32),
                  "labels": torch.tensor([2, 1])}
        prediction = {"boxes": torch.tensor([[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30], [20, 20, 30, 30]], dtype=torch.float32),
                      "labels": torch.tensor([2, 2, 3, 1]), "scores": torch.tensor([.9, .8, .7, .4])}
        matches = match_predictions(prediction, target)
        self.assertEqual(matches, {"correct": [0], "incorrect": [1, 2], "missed": [1]})
        summary, _ = summarize([prediction], [target], .5)
        self.assertEqual(summary["overall"]["tp"], 1)
        self.assertEqual(summary["overall"]["fp"], 2)
        self.assertEqual(summary["overall"]["fn"], 1)
        self.assertEqual(summary["per_class"]["Short"]["precision"], .5)
        self.assertEqual(match_predictions(prediction, target, .3)["missed"], [])

    def test_empty_predictions(self):
        prediction = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.int64), "scores": torch.empty(0)}
        target = {"boxes": torch.tensor([[0., 0., 5., 5.]]), "labels": torch.tensor([2])}
        self.assertEqual(match_predictions(prediction, target)["missed"], [0])


if __name__ == "__main__":
    unittest.main()
