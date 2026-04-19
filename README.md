# Real-Time Action Recognition via 3D CNN and Optical Flow Fusion

**CS 5330 — Pattern Recognition and Computer Vision**  
**Group members:** Shenwei Fan, Siwei An

---

## Project Description

We build a real-time action recognition system on UCF-101 using a two-branch fusion
architecture: a 3D CNN (C3D) branch for spatiotemporal RGB features, and an Optical
Flow + ResNet-18 branch for explicit motion cues. We evaluate Top-1 / Top-5 accuracy
and inference FPS, and compare three configurations via ablation study:
3D-CNN only, Optical-Flow only, and the full fusion model.

---

## Repository Structure

```
Final/
├── config.py             # all hyperparameters and path settings
├── dataset.py            # UCF-101 clip sampling and data loaders
├── train.py              # training loop (Person A)
├── inference_fps.py      # FPS benchmark (Person A)
├── models/
│   ├── __init__.py
│   ├── c3d.py            # C3D 3D-CNN implementation
│   └── fusion.py         # FusionModel (3d_only / fusion modes)
├── splits/               # UCF-101 train/test split text files
├── data/
│   └── UCF-101/          # video files  <ClassName>/<video>.avi
└── checkpoints/          # saved during training (auto-created)
```

---

## Setup

### 1. Install dependencies (GPU — CUDA 12.1)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Confirm data layout
```
data/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
data/UCF-101/...
splits/trainlist01.txt
splits/testlist01.txt
```

---

## Training (Person A — 3D CNN only)

```bash
python train.py
```

Resume from a checkpoint:
```bash
python train.py --resume checkpoints/last.pth
```

Training logs Top-1 / Top-5 accuracy each epoch.  
Best checkpoint saved to `checkpoints/best.pth`.

---

## Inference FPS Benchmark

```bash
python inference_fps.py --ckpt checkpoints/best.pth
```

---

## Configuration (`config.py`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `NUM_CLASSES` | 101 | set 10 + `SUBSET=True` for CPU debug |
| `CLIP_LEN` | 16 | frames per clip |
| `FRAME_SIZE` | 112 | spatial resolution |
| `BATCH_SIZE` | 8 | reduce to 4 if GPU OOM |
| `NUM_EPOCHS` | 30 | |
| `LR` | 1e-3 | SGD with StepLR ÷10 every 10 epochs |
| `FUSION_MODE` | `"3d_only"` | change to `"fusion"` after Person B integrates |
| `SUBSET` | `False` | `True` → 10-class subset for CPU testing |

---

## Integrating Person B's Optical Flow Branch

Once Person B's branch is ready:

1. Set `FUSION_MODE = "fusion"` in `config.py`
2. Update `other_feat_dim` in `models/fusion.py` to match Person B's output size
3. Pass Person B's feature vector during training:
   ```python
   logits = model(x_3d, x_other=feat_b)
   ```

---

## Demo / Data

- Video demo: [TODO — add Google Drive / YouTube URL]
- Dataset: UCF-101 — https://www.crcv.ucf.edu/data/UCF101.php
