"""
Inference FPS benchmark for the C3D / FusionModel.

Measures forward-pass throughput using random tensors (no I/O overhead).

Usage:
    python inference_fps.py                          # random weights
    python inference_fps.py --ckpt checkpoints/best.pth
"""

import argparse
import time
import torch
import config
from models import FusionModel


def benchmark(model, device, batch_size, num_runs=200, warmup=20):
    model.eval()
    dummy = torch.randn(
        batch_size, 3, config.CLIP_LEN, config.FRAME_SIZE, config.FRAME_SIZE,
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

    elapsed = time.perf_counter() - t0
    fps        = (num_runs * batch_size) / elapsed
    ms_per_clip = elapsed * 1000 / (num_runs * batch_size)
    return fps, ms_per_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="Path to checkpoint (.pth)")
    args = parser.parse_args()

    model = FusionModel(num_classes=config.NUM_CLASSES, mode=config.FUSION_MODE)

    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
        print(f"Loaded: {args.ckpt}  "
              f"(epoch={state.get('epoch','?')}, top1={state.get('top1',0):.2f}%)")
    else:
        print("No checkpoint — using random weights.")

    print("=" * 55)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model_gpu = model.to(device)
        for bs in [1, 8]:
            fps, ms = benchmark(model_gpu, device, batch_size=bs)
            print(f"[GPU] batch={bs:2d} | {fps:8.1f} FPS  ({ms:.2f} ms/clip)")
    else:
        print("[GPU] CUDA not available, skipping.")

    device_cpu = torch.device("cpu")
    model_cpu  = model.to(device_cpu)
    fps, ms    = benchmark(model_cpu, device_cpu, batch_size=1, num_runs=20, warmup=3)
    print(f"[CPU] batch= 1 | {fps:8.1f} FPS  ({ms:.2f} ms/clip)")

    print("=" * 55)


if __name__ == "__main__":
    main()
