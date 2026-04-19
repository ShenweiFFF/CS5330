import os

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT   = os.path.join("data", "UCF-101")
SPLITS_DIR  = "splits"
CKPT_DIR    = "checkpoints"

# ─── Dataset ──────────────────────────────────────────────────────────────────
NUM_CLASSES  = 101
SPLIT_ID     = 1          # UCF-101 provides 3 splits; use split 1
SUBSET       = False      # set True to use 10-class subset (CPU debug only)

# ─── Clip sampling ────────────────────────────────────────────────────────────
CLIP_LEN     = 16         # frames per clip
FRAME_SIZE   = 112        # spatial size (H = W)
STRIDE       = 1          # temporal stride when sampling frames

# ─── Normalization (Kinetics mean/std) ────────────────────────────────────────
MEAN = [0.43216, 0.394666, 0.37645]
STD  = [0.22803, 0.22145, 0.216989]

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 8
NUM_WORKERS     = 4
NUM_EPOCHS      = 30
LR              = 1e-3
LR_DECAY_STEP   = 10      # StepLR: decay every N epochs
LR_DECAY_GAMMA  = 0.1
WEIGHT_DECAY    = 5e-4

# ─── Model ────────────────────────────────────────────────────────────────────
# fusion_mode: "3d_only" | "fusion"
# Use "3d_only" until Person B's optical flow branch is integrated
FUSION_MODE  = "3d_only"
DROPOUT      = 0.5
