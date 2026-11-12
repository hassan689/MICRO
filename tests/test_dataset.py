"""Exercise detection data shapes and invalid annotation handling."""

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from preprocessing.dataset import DeepPCBDataset, detection_collate


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "trainval.txt").write_text("one.jpg one.txt\ntwo.jpg two.txt\n")
        Image.new("L", (32, 24), 255).save(self.root / "one_test.jpg")
        Image.new("RGB", (40, 30)).save(self.root / "two_test.jpg")
        (self.root / "one.txt").write_text("1 2 10 12 6\n")
        (self.root / "two.txt").write_text("")

    def test_rgb_tensor_and_detection_target(self):
        image, target = DeepPCBDataset(self.root)[0]
        self.assertEqual(tuple(image.shape), (3, 24, 32))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(float(image.max()), 1.0)
        self.assertEqual(target["boxes"].tolist(), [[1, 2, 10, 12]])
        self.assertEqual(target["boxes"].dtype, torch.float32)
        self.assertEqual(target["labels"].tolist(), [6])
        self.assertEqual(target["labels"].dtype, torch.int64)
        self.assertEqual(target["area"].tolist(), [90])

    def test_variable_images_and_empty_boxes_batch(self):
        loader = DataLoader(DeepPCBDataset(self.root), batch_size=2,
                            collate_fn=detection_collate)
        images, targets = next(iter(loader))
        self.assertEqual(len(images), 2)
        self.assertNotEqual(images[0].shape, images[1].shape)
        self.assertEqual(tuple(targets[1]["boxes"].shape), (0, 4))
        self.assertEqual(tuple(targets[1]["labels"].shape), (0,))

    def test_reject_invalid_annotations(self):
        for annotation in ("1 2 10 12 0", "10 2 1 12 1", "1 2 33 12 1", "bad"):
            with self.subTest(annotation=annotation):
                (self.root / "one.txt").write_text(annotation)
                with self.assertRaises(ValueError):
                    DeepPCBDataset(self.root)[0]

    def test_missing_image(self):
        (self.root / "one_test.jpg").unlink()
        with self.assertRaises(FileNotFoundError):
            DeepPCBDataset(self.root)


if __name__ == "__main__":
    unittest.main()
