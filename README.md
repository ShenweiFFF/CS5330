# Real-Time Action Recognition via 3D CNN and Optical Flow Fusion

**CS 5330 — Pattern Recognition and Computer Vision**  
**Group members:** Shenwei Fan (RGB branch), Siwei An (Flow branch)

---

## Project Description

We build a real-time action recognition system on UCF-101 using a two-branch
fusion architecture: a **3D ResNet-18** branch for spatiotemporal RGB features,
and a **Farneback Optical Flow + 2D ResNet-18** branch for explicit motion cues.
Feature vectors (512-d each) from both branches are concatenated and classified
by a fully-connected fusion head (FC-512 → ReLU → Dropout → FC-101).

The system achieves **87.2% Top-1 accuracy at 22.8 FPS** on an NVIDIA RTX 5060 Ti
using FP16 inference and 8-frame subsampling, exceeding the real-time target of ≥15 FPS.

---

## Repository Structure

```
Final/
├── config.py                  # shared hyperparameters and path settings
├── dataset.py                 # UCF-101 RGB clip sampling and DataLoader
├── train.py                   # RGB branch training (Person A)
├── inference_fps.py           # FPS benchmark — full-accuracy & real-time modes
├── models/
│   ├── __init__.py
│   ├── r3d18.py               # 3D ResNet-18 (512-d spatiotemporal features)
│   └── fusion.py              # FusionModel: rgb_only | fusion
├── flow_branch/               # Optical flow branch (Person B)
│   ├── extract_flow.cpp       # C++: Farneback flow extraction → .yml files
│   ├── flow_model.hpp         # C++ LibTorch: FlowResNet18 + Dataset
│   ├── train_eval.cpp         # C++: train / eval / ablation (LibTorch)
│   ├── train_flow_ucf101.py   # Python: flow branch training (PyTorch)
│   ├── CMakeLists.txt         # CMake build for C++ tools
│   └── ucf101_flow/           # precomputed flow maps (not in repo — generate locally)
├── splits/                    # UCF-101 train/test split text files
├── data/
│   └── UCF-101/               # raw video files  <ClassName>/<video>.avi
└── checkpoints/               # saved during training (auto-created)
```

---

## Demo / Data

- **Video demo:** [TODO — add YouTube / Google Drive URL before submission]
- **Dataset:** UCF-101 — https://www.crcv.ucf.edu/data/UCF101.php

> **Note on checkpoints:** The trained checkpoints (best.pth / last.pth) use the
> 3D ResNet-18 (R3D18) architecture. Any older `.pth` files saved with the
> original C3D backbone are **not compatible** and should not be used with
> `--resume` or `--ckpt`.

---

## Setup

### 1. Install Python dependencies (CUDA 12.1)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python numpy
```

### 2. Confirm data layout
```
data/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
splits/trainlist01.txt
splits/testlist01.txt
```

---

## Step 1 — Extract Optical Flow (Person B)

Build and run the C++ flow extractor (requires OpenCV):

```bash
cd flow_branch
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build . --config Release

# Run from flow_branch/ — reads data/UCF-101, writes flow_branch/ucf101_flow/
cd ..
./build/extract_flow
```

This produces `flow_branch/ucf101_flow/<ClassName>/<clip_stem>/flow_000.yml … flow_015.yml`.

---

## Step 2 — Train RGB Branch (Person A)

```bash
python train.py                              # train from scratch (60 epochs)
python train.py --resume checkpoints/last.pth   # resume
```

Training logs Top-1 / Top-5 accuracy each epoch.  
Best checkpoint → `checkpoints/best.pth`.

---

## Step 3 — Train Flow Branch (Person B)

```bash
cd flow_branch
python train_flow_ucf101.py --split 1
```

Or use the C++ trainer (requires LibTorch):
```bash
./build/train_eval --split 1
```

Best checkpoint → `flow_branch/flow_branch_split1.pt`.

---

## Step 4 — Inference FPS Benchmark

```bash
python inference_fps.py --ckpt checkpoints/best.pth
```

Reports throughput for both operating points:

| Mode            | Frames | Resolution | Description                        |
|-----------------|--------|------------|------------------------------------|
| Full-accuracy   | 16     | 224×224    | Training resolution                |
| Real-time       | 8      | 112×112    | 8-frame subsampling (≥15 FPS)      |

---

## Configuration (`config.py`)

| Parameter              | Default          | Notes                                        |
|------------------------|------------------|----------------------------------------------|
| `NUM_CLASSES`          | 101              | set `SUBSET=True` for 10-class CPU debug     |
| `CLIP_LEN`             | 16               | frames per training clip                     |
| `FRAME_SIZE`           | 224              | training spatial resolution                  |
| `INFER_CLIP_LEN`       | 8                | frames for real-time inference               |
| `INFER_FRAME_SIZE`     | 112              | resolution for real-time inference           |
| `BATCH_SIZE`           | 32               | reduce to 8 if GPU OOM                       |
| `NUM_EPOCHS`           | 60               |                                              |
| `LR`                   | 1e-3             | SGD; ×0.1 at epochs 30 and 45               |
| `FUSION_MODE`          | `"rgb_only"`     | change to `"fusion"` for two-branch mode     |

---

## Fusion Architecture (paper §III.A)

```
RGB clip (B, 3, 16, 224, 224)
    └─ R3D18 (feature_only=True)  ──► 512-d ─┐
                                               ├─ concat (B, 1024)
Flow stack (B, 32, 224, 224)                   │        │
    └─ FlowResNet18 (avgpool out) ──► 512-d ──┘         ▼
                                               FC-512 → ReLU → Dropout(0.5) → FC-101
```
