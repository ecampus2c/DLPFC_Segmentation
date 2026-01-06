import os

# --- PATHS ---
# Update DATA_ROOT to your local path or mount point
DATA_ROOT = 'dataset' 
IMG_DIR = os.path.join(DATA_ROOT, 'images')
LBL_DIR = os.path.join(DATA_ROOT, 'labels')

# Output directories
PREPROC_DIR = 'preproc'
MODELS_DIR = 'models'
PREDS_DIR = 'preds'
LOGS_DIR = 'logs'

# Create dirs if they don't exist
for d in [PREPROC_DIR, MODELS_DIR, PREDS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- HYPERPARAMETERS ---
TARGET_SPACING = (1.0, 1.0, 1.0)
TARGET_SHAPE = (128, 128, 128)
PATCH_SIZE = (96, 96, 96)
PATCH_STRIDE = (48, 48, 48)  # 50% overlap

BATCH_SIZE = 1
BASE_FILTERS = 16
EPOCHS = 200
STEPS_PER_EPOCH = 200
VAL_PATCHES = 128
PATCHES_PER_SUBJECT = 256
POS_NEG_RATIO = 0.6
LR = 1e-4
SEED = 42