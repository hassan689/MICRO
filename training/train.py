import argparse
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from evaluation.metrics import evaluate
from models.detector import build_detector
from preprocessing.dataset import CLASS_NAMES, DEFAULT_DATA_ROOT, PROJECT_ROOT, DeepPCBDataset, detection_collate
from preprocessing.splits import DEFAULT_SPLIT_PATH, load_or_create_split


def save_checkpoint(checkpoint, path):
    temporary = path.with_suffix(".pt.tmp")
    with temporary.open("wb") as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for step, (images, targets) in enumerate(loader, 1):
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        optimizer.zero_grad(set_to_none=True)
        losses = model(images, targets)
        loss = sum(losses.values())
        if not math.isfinite(float(loss.detach())):
            raise RuntimeError(f"Non-finite loss at batch {step}: {losses}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
        total_loss += float(loss.detach())
        if step == 1 or step % 10 == 0 or step == len(loader):
            print(f"  training batch {step}/{len(loader)}: loss={float(loss.detach()):.4f}", flush=True)
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="Continue a checkpoint in a new output directory; epochs is the total target")
    parser.add_argument("--no-pretrained", action="store_true", help="Random initialization; no downloads")
    parser.add_argument("--smoke-test", action="store_true", help="One epoch on 2 train and 2 validation images at 320px")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs, batch size, threads, and learning rate must be positive")
    if args.image_size < 64:
        parser.error("image size must be at least 64")
    if args.smoke_test:
        args.epochs, args.batch_size, args.image_size = 1, 2, 320
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this environment. Use --device cpu or install a compatible GPU build.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    dataset = DeepPCBDataset(args.root)
    manifest = load_or_create_split(dataset, args.split_file, args.val_fraction, args.seed)
    resume = None
    best_checkpoint = None
    start_epoch = 1
    if args.resume:
        if args.smoke_test:
            parser.error("Resume cannot be combined with smoke-test")
        resume = torch.load(args.resume, map_location="cpu", weights_only=True)
        start_epoch = resume["epoch"] + 1
        if start_epoch > args.epochs:
            parser.error("Checkpoint already reached the requested total epochs")
        for key in ("batch_size", "image_size", "lr", "seed", "no_pretrained"):
            if resume["config"][key] != getattr(args, key):
                parser.error(f"Resume requires the original {key}: {resume['config'][key]}")
        previous_split = json.loads((args.resume.parent / "split.json").read_text(encoding="utf-8"))
        if previous_split != manifest:
            parser.error("Resume dataset/split does not match the saved run")
        previous_source = (args.resume.parent / "dataset_source.py").read_text(encoding="utf-8")
        if previous_source != (PROJECT_ROOT / "preprocessing/dataset.py").read_text(encoding="utf-8"):
            parser.error("Dataset preprocessing changed since the saved run")
        if resume["metrics"]["map50"] == resume["best_map50"]:
            best_checkpoint = resume
        else:
            best_checkpoint = torch.load(args.resume.parent / "best.pt", map_location="cpu", weights_only=True)
            if best_checkpoint["metrics"]["map50"] != resume["best_map50"]:
                parser.error("Saved best checkpoint does not match the resumed run")
    train_ids, val_ids = manifest["train_indices"], manifest["val_indices"]
    if args.smoke_test:
        train_ids, val_ids = train_ids[:2], val_ids[:2]
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(Subset(dataset, train_ids), batch_size=args.batch_size,
                              shuffle=True, generator=generator, num_workers=0,
                              collate_fn=detection_collate)
    val_loader = DataLoader(Subset(dataset, val_ids), batch_size=args.batch_size,
                            num_workers=0, collate_fn=detection_collate)
    kind = "smoke" if args.smoke_test else "run"
    output = args.output or PROJECT_ROOT / "outputs/training" / f"{kind}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}. Choose a new --output.")
    output.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(PROJECT_ROOT / "checkpoints/torch_hub"))
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(device=str(device), output=str(output), train_samples=len(train_ids),
                  val_samples=len(val_ids), torch_version=str(torch.__version__),
                  torchvision_version=str(torchvision.__version__),
                  architecture="fasterrcnn_mobilenet_v3_large_320_fpn",
                  preprocessing="RGB + existing enhance_image in preprocessing/dataset.py")
    # Preserve the exact loader used, including the user's enhancement settings.
    loader_source = (PROJECT_ROOT / "preprocessing/dataset.py").read_text(encoding="utf-8")
    (output / "dataset_source.py").write_text(loader_source, encoding="utf-8")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output / "split.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Device: {device} | Training: {len(train_ids)} | Validation: {len(val_ids)}", flush=True)
    print(f"Output: {output}", flush=True)
    if args.smoke_test:
        print("SMOKE TEST: these checkpoints and metrics are only for verifying the pipeline.", flush=True)
    # Keep the original backbone freezing policy so optimizer groups match on resume.
    model = build_detector(not args.no_pretrained, args.image_size).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr, momentum=0.9, weight_decay=0.0005)
    best = -1.0
    if resume is not None:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        best = resume["best_map50"]
        save_checkpoint(best_checkpoint, output / "best.pt")
        if "rng_state" in resume:
            torch.set_rng_state(resume["rng_state"])
            generator.set_state(resume["loader_rng_state"])
            random.setstate(resume["python_rng_state"])
            if device.type == "cuda" and resume.get("cuda_rng_states"):
                torch.cuda.set_rng_state_all(resume["cuda_rng_states"])
        else:
            # Older checkpoints did not save random state; use a fresh, repeatable shuffle.
            torch.manual_seed(args.seed + resume["epoch"])
            random.seed(args.seed + resume["epoch"])
            generator.manual_seed(args.seed + resume["epoch"])
            print("Older checkpoint: model and optimizer restored; random sampling is reseeded.", flush=True)
        print(f"Resuming after epoch {resume['epoch']} | Best AP50: {best:.4f}", flush=True)
        del resume, best_checkpoint
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        loss = train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        improved = metrics["map50"] > best
        best = max(best, metrics["map50"])
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "epoch": epoch, "best_map50": best, "metrics": metrics,
                      "config": config, "classes": CLASS_NAMES,
                      "rng_state": torch.get_rng_state(), "loader_rng_state": generator.get_state(),
                      "python_rng_state": random.getstate(),
                      "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        save_checkpoint(checkpoint, output / "last.pt")
        if improved:
            save_checkpoint(checkpoint, output / "best.pt")
        record = {"epoch": epoch, "train_loss": loss, "validation": metrics,
                  "seconds": time.perf_counter() - started}
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"Loss: {loss:.4f} | Validation AP50: {metrics['map50']:.4f} | {record['seconds']:.1f}s", flush=True)
    print(f"Finished. Best checkpoint: {output / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
