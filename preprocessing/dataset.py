import argparse
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image, to_tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/raw/DeepPCB/PCBData"
CLASS_NAMES = {
    1: "Open", 2: "Short", 3: "Mousebite",
    4: "Spur", 5: "Copper", 6: "Pin hole",
}
# I tried to use the original image but the visualization was too dark, so I applied a contrast enhancement to make the defects more visible.
def enhance_image(image):
    # better visualization without changing the orgianal image
    image = ImageEnhance.Contrast(image).enhance(1.25)
    image = ImageEnhance.Sharpness(image).enhance(1.4)
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=3))
    return image

class DeepPCBDataset(Dataset):
    def __init__(self, root=DEFAULT_DATA_ROOT, split="trainval"):
        if split not in {"trainval", "test"}:
            raise ValueError("split must be 'trainval' or 'test'")

        self.root = Path(root)
        self.samples = []
        split_path = self.root / f"{split}.txt"

        for line_number, line in enumerate(split_path.read_text().splitlines(), 1):
            if not line.strip():
                continue

            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"{split_path}:{line_number}: expected two paths")

            base = Path(fields[0])
            image_path = self.root / base.with_name(f"{base.stem}_test.jpg")
            label_path = self.root / fields[1]

            for path in (image_path, label_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing dataset file: {path}")

            self.samples.append((image_path, label_path))

        if not self.samples:
            raise ValueError(f"No samples in {split_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        image = enhance_image(image)
        width, height = image.size
        boxes, labels = [], []
        for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                x1, y1, x2, y2, class_id = map(int, line.split())
            except ValueError as error:
                raise ValueError(
                    f"{label_path}:{line_number}: expected five integers"
                ) from error
            if class_id not in CLASS_NAMES:
                raise ValueError(f"{label_path}:{line_number}: invalid class {class_id}")
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"{label_path}:{line_number}: invalid bounding box")
            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)
        """here we are converting the boxes and labels to tensors and creating a target dictionary that contains the bounding boxes, labels, image ID, area of the boxes, and a flag indicating whether the boxes are crowded or not. This target dictionary will be used for training object detection models."""
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": boxes,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }

        return to_tensor(image), target


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", choices=["trainval", "test"], default="trainval")
    parser.add_argument("--check-all", action="store_true", help="Read every sample")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs/dataset_preview.jpg",
        help="Where to save the first image with ground-truth boxes",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    dataset = DeepPCBDataset(args.root, args.split)
    loader = DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=detection_collate)
    images, targets = next(iter(loader))
    print(f"Split: {args.split} | Samples: {len(dataset)}")
    print(f"First batch: {len(images)} images")
    print(f"Image shape: {tuple(images[0].shape)} | dtype: {images[0].dtype}")
    print(f"First image labels: {targets[0]['labels'].tolist()}")
    preview = to_pil_image(images[0])
    draw = ImageDraw.Draw(preview)
    for box, label in zip(targets[0]["boxes"].tolist(), targets[0]["labels"].tolist()):
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], max(0, box[1] - 12)), CLASS_NAMES[label], fill="red")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    preview.save(args.output)
    print(f"Preview saved: {args.output.resolve()}")
    if args.check_all:
        count = 0
        for index in range(len(dataset)):
            _, target = dataset[index]
            count += len(target["labels"])
        print(f"Verified {len(dataset)} images and {count} defect boxes.")


if __name__ == "__main__":
    main()
