"""
Training script for the RGB branch (R3D18) on UCF-101.

Supports "rgb_only" mode (standalone) and "fusion" mode once Person B's
precomputed flow features are integrated.  See config.FUSION_MODE.

Usage:
    python train.py                              # train from scratch
    python train.py --resume checkpoints/last.pth   # resume training
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import config
from dataset import get_loaders
from models import FusionModel


def accuracy(output: torch.Tensor, target: torch.Tensor,
             topk: tuple[int, ...] = (1, 5)) -> list[float]:
    with torch.no_grad():
        maxk  = max(topk)
        batch = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred    = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [
            correct[:k].reshape(-1).float().sum().mul_(100.0 / batch).item()
            for k in topk
        ]


def train_one_epoch(model: nn.Module, loader, criterion: nn.Module,
                    optimizer: optim.Optimizer, device: torch.device,
                    epoch: int) -> None:
    model.train()
    total_loss = total_top1 = total_top5 = 0.0
    t0 = time.time()

    for i, (clips, labels) in enumerate(loader):
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(clips)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        top1, top5 = accuracy(logits, labels)
        total_loss += loss.item()
        total_top1 += top1
        total_top5 += top5

        if (i + 1) % 50 == 0:
            print(f"  [Epoch {epoch} | Step {i+1}/{len(loader)}] "
                  f"loss={loss.item():.4f}  top1={top1:.1f}%  top5={top5:.1f}%")

    n = len(loader)
    print(f"Epoch {epoch:02d} TRAIN | "
          f"loss={total_loss/n:.4f}  top1={total_top1/n:.2f}%  "
          f"top5={total_top5/n:.2f}%  ({time.time()-t0:.0f}s)")


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module,
             device: torch.device, epoch: int) -> float:
    model.eval()
    total_loss = total_top1 = total_top5 = 0.0

    for clips, labels in loader:
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(clips)
        loss   = criterion(logits, labels)
        top1, top5 = accuracy(logits, labels)
        total_loss += loss.item()
        total_top1 += top1
        total_top5 += top5

    n = len(loader)
    print(f"Epoch {epoch:02d} EVAL  | "
          f"loss={total_loss/n:.4f}  top1={total_top1/n:.2f}%  "
          f"top5={total_top5/n:.2f}%")
    return total_top1 / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    os.makedirs(config.CKPT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  mode: {config.FUSION_MODE}")

    train_loader, test_loader = get_loaders()
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    model = FusionModel(num_classes=config.NUM_CLASSES, mode=config.FUSION_MODE)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LR,
        momentum=0.9,
        weight_decay=config.WEIGHT_DECAY,
    )
    # MultiStepLR: decay LR ×0.1 at epochs 30 and 45 (paper §IV.D)
    scheduler = MultiStepLR(
        optimizer,
        milestones=config.LR_DECAY_MILESTONES,
        gamma=config.LR_DECAY_GAMMA,
    )

    start_epoch = 1
    best_top1   = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_top1   = ckpt.get("top1", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']} (best top-1: {best_top1:.2f}%)")

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        top1 = evaluate(model, test_loader, criterion, device, epoch)
        scheduler.step()

        state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "top1":      top1,
        }
        torch.save(state, os.path.join(config.CKPT_DIR, "last.pth"))

        if top1 > best_top1:
            best_top1 = top1
            torch.save(state, os.path.join(config.CKPT_DIR, "best.pth"))
            print(f"  *** New best top-1: {best_top1:.2f}% ***")

    print(f"\nTraining complete. Best top-1: {best_top1:.2f}%")


if __name__ == "__main__":
    main()
