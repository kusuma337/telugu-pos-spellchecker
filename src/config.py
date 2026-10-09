"""Project-wide configuration. Paths are relative to the project root."""

from __future__ import annotations

from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Project root resolution
#
# When this file is executed from C:\Users\K.JAGADHISH\Desktop\telugu-pos-spellchecker
# (the tool workspace mirror), __file__ resolves to the Desktop path, which has
# no models/ folder.  The real project — and trained artifacts — live at:
#   C:\Users\K.JAGADHISH\telugu-pos-spellchecker
#
# Strategy: start from __file__ and walk up until we find a directory that
# contains models/bilstm_pos.pt.  If no such ancestor exists (fresh clone,
# CI, etc.) fall back to the immediate parent.parent as before.
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Return the nearest ancestor directory that contains models/bilstm_pos.pt."""
    candidate = Path(__file__).resolve().parent.parent
    # Check up to 3 levels above the file's own parent.parent
    for _ in range(4):
        if (candidate / "models" / "bilstm_pos.pt").exists():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    # Fallback: sibling directory with the same folder name but without "Desktop\"
    default = Path(__file__).resolve().parent.parent
    # Try stripping "Desktop\" from the path if present
    parts = default.parts
    if "Desktop" in parts:
        idx = parts.index("Desktop")
        # Rebuild path without the Desktop segment
        non_desktop = Path(*parts[:idx]) / parts[idx + 1] if idx + 1 < len(parts) else default
        if (non_desktop / "models" / "bilstm_pos.pt").exists():
            return non_desktop
    return default


PROJECT_ROOT = _find_project_root()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

SAMPLE_DATA_FILENAME = "sample_telugu_pos.txt"
SAMPLE_DATA_PATH = RAW_DATA_DIR / SAMPLE_DATA_FILENAME

MODEL_PATH = MODELS_DIR / "bilstm_pos.pt"
VOCAB_PATH = MODELS_DIR / "vocabulary.json"
SPELLCHECKER_PATH = MODELS_DIR / "spellchecker.json"
TRAIN_HISTORY_PATH = METRICS_DIR / "training_history.json"
EVAL_METRICS_PATH = METRICS_DIR / "evaluation_metrics.json"

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_TAG = "<PAD>"

SEED = 42

EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.3
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MIN_WORD_FREQ = 1
MAX_SEQ_LEN = 64
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
EARLY_STOPPING_PATIENCE = 6
GRAD_CLIP_NORM = 5.0

SPELLCHECK_MAX_EDIT_DISTANCE = 2
SPELLCHECK_MIN_FREQ = 1
SPELLCHECK_THRESHOLD = -12.0
SPELLCHECK_EDIT1_LOG_PROB = -1.0
SPELLCHECK_EDIT2_LOG_PROB = -3.0
SPELLCHECK_KEEP_OOV_LOG_PROB = -8.0


def get_device() -> torch.device:
    """Use CUDA when available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_project_dirs() -> None:
    """Create output directories if they do not exist."""
    for path in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
