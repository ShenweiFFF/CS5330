import os

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT  = os.path.join("data", "UCF-101")   # raw video files
SPLITS_DIR = "splits"                           # train/test split text files
CKPT_DIR   = "checkpoints"                      # saved model checkpoints
FLOW_ROOT  = os.path.join("flow_branch", "ucf101_flow")  # precomputed flow maps

# ─── Dataset ──────────────────────────────────────────────────────────────────
NUM_CLASSES = 101
SPLIT_ID    = 1       # UCF-101 provides 3 splits; paper reports mean over all 3
SUBSET      = False   # True → 10-class subset for fast CPU debugging

# ─── Clip sampling (training / full-accuracy evaluation) ──────────────────────
CLIP_LEN    = 16      # frames per clip (matches paper §III.A)
FRAME_SIZE  = 224     # spatial resolution H = W (matches paper §III.A)
STRIDE      = 1       # temporal stride when sampling frames

# ─── Real-time inference settings (§III.C) ────────────────────────────────────
INFER_CLIP_LEN    = 8    # 8-frame subsampling → ≥15 FPS on RTX GPU
INFER_FRAME_SIZE  = 112  # downsampled resolution for real-time deployment

# ─── Normalization (ImageNet mean/std — both branches are ImageNet-pretrained) ─
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE           = 32
NUM_WORKERS          = 4
NUM_EPOCHS           = 60
LR                   = 1e-3
LR_DECAY_MILESTONES  = [30, 45]   # MultiStepLR: ×0.1 at these epoch boundaries
LR_DECAY_GAMMA       = 0.1
WEIGHT_DECAY         = 5e-4

# ─── Model ────────────────────────────────────────────────────────────────────
# "rgb_only" — RGB branch (R3D18) only; default for standalone training
# "fusion"   — full two-branch fusion; requires precomputed flow features
FUSION_MODE = "rgb_only"
DROPOUT     = 0.5
