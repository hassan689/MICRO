"""Class-wise all-points interpolated AP at IoU 0.5 (not COCO mAP)."""

import torch
from torchvision.ops import box_iou

from preprocessing.dataset import CLASS_NAMES


def detection_metrics(predictions, targets, iou_threshold=0.5):
    if len(predictions) != len(targets):
        raise ValueError("Each prediction must have a matching target")
    per_class = {}
    for class_id, name in CLASS_NAMES.items():
        ground_truth = [t["boxes"][t["labels"] == class_id].cpu() for t in targets]
        total = sum(len(boxes) for boxes in ground_truth)
        candidates = []
        for image_id, prediction in enumerate(predictions):
            mask = prediction["labels"] == class_id
            for score, box in zip(prediction["scores"][mask], prediction["boxes"][mask]):
                candidates.append((float(score), image_id, box.cpu()))
        candidates.sort(key=lambda item: item[0], reverse=True)
        matched = [set() for _ in targets]
        tp, fp = [], []
        for _, image_id, box in candidates:
            boxes = ground_truth[image_id]
            correct = False
            if len(boxes):
                overlaps = box_iou(box.unsqueeze(0), boxes)[0]
                overlap, match = overlaps.max(dim=0)
                match = int(match)
                if float(overlap) >= iou_threshold and match not in matched[image_id]:
                    matched[image_id].add(match)
                    correct = True
            tp.append(int(correct))
            fp.append(int(not correct))
        ap = None
        if total:
            true_positives = torch.tensor(tp, dtype=torch.float64).cumsum(0)
            false_positives = torch.tensor(fp, dtype=torch.float64).cumsum(0)
            recall = true_positives / total
            precision = true_positives / (true_positives + false_positives).clamp(min=1)
            recall = torch.cat((torch.tensor([0.]), recall, torch.tensor([1.])))
            precision = torch.cat((torch.tensor([0.]), precision, torch.tensor([0.])))
            precision = precision.flip(0).cummax(0).values.flip(0)
            ap = float(((recall[1:] - recall[:-1]) * precision[1:]).sum())
        per_class[name] = {"ap50": ap, "ground_truth": total,
                           "predictions": len(candidates)}
    values = [item["ap50"] for item in per_class.values() if item["ap50"] is not None]
    return {"map50": sum(values) / len(values) if values else 0.0,
            "images": len(targets), "per_class": per_class}


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    predictions, targets = [], []
    for index, (images, batch_targets) in enumerate(loader, 1):
        outputs = model([image.to(device) for image in images])
        predictions.extend({key: value.cpu() for key, value in output.items()}
                           for output in outputs)
        targets.extend(batch_targets)
        if index % 25 == 0:
            print(f"  validation batch {index}/{len(loader)}", flush=True)
    return detection_metrics(predictions, targets)
