"""
Inference FPS benchmark for the RGB branch (R3D18 / FusionModel).

Two operating points are measured (paper §III.C):

  Full-accuracy mode  — 16 frames, 224×224  (training resolution)
  Real-time mode      — 8 frames,  112×112  (8-frame subsampling + half resolution)

Throughput is measured using random tensors to isolate model latency from I/O.

Usage:
    python inference_fps.py                          # random weights
    python inference_fps.py --ckpt checkpoints/best.pth
"""

import argparse
import time

import torch

import config
from models import FusionModel


def benchmark(model: torch.nn.Module, device: torch.device,
              clip_len: int, frame_size: int,
              batch_size: int = 1,
              num_runs: int = 200, warmup: int = 20) -> tuple[float, float]:
    model.eval()
    dummy = torch.randn(
        batch_size, 3, clip_len, frame_size, frame_size,
        device=device,
    )

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed     = time.perf_counter() - t0
    fps         = (num_runs * batch_size) / elapsed
    ms_per_clip = elapsed * 1000 / (num_runs * batch_size)
    return fps, ms_per_clip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None,
                        help="Path to checkpoint (.pth)")
    args = parser.parse_args()

    model = FusionModel(num_classes=config.NUM_CLASSES, mode="rgb_only")

    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
        print(f"Loaded: {args.ckpt}  "
              f"(epoch={state.get('epoch','?')}, top1={state.get('top1',0):.2f}%)")
    else:
        print("No checkpoint — using random weights.")

    print("=" * 65)

    configs = [
        ("Full-accuracy", config.CLIP_LEN,       config.FRAME_SIZE),
        ("Real-time",     config.INFER_CLIP_LEN,  config.INFER_FRAME_SIZE),
    ]

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model_gpu = model.to(device)
        for label, clip_len, frame_size in configs:
            for bs in [1, 8]:
                fps, ms = benchmark(model_gpu, device, clip_len, frame_size,
                                    batch_size=bs)
                print(f"[GPU] {label:15s} | T={clip_len:2d} {frame_size}×{frame_size} "
                      f"batch={bs:2d} | {fps:8.1f} FPS  ({ms:.2f} ms/clip)")
    else:
        print("[GPU] CUDA not available, skipping.")

    device_cpu = torch.device("cpu")
    model_cpu  = model.to(device_cpu)
    for label, clip_len, frame_size in configs:
        fps, ms = benchmark(model_cpu, device_cpu, clip_len, frame_size,
                            batch_size=1, num_runs=10, warmup=2)
        print(f"[CPU] {label:15s} | T={clip_len:2d} {frame_size}×{frame_size} "
              f"batch= 1 | {fps:8.1f} FPS  ({ms:.2f} ms/clip)")

    print("=" * 65)


if __name__ == "__main__":
    main()
