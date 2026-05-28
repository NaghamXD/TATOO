from pathlib import Path

ROOT = Path(__file__).parent

# ── Data paths ────────────────────────────────────────────────────────────────
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR   = ROOT / "results"

DEMOGRAPHICS_FILE = ROOT / "data" / "raw" / "demographicsv2.csv"

# ── Games ─────────────────────────────────────────────────────────────────────
TAPPING_GAMES = ["TouchIt", "CornerIt", "DoubleTapIt"]
DRAGGING_GAMES = ["PinchIt", "SlideIt", "DragIt"]
ALL_GAMES = TAPPING_GAMES + DRAGGING_GAMES

FIXATION_GAMES = ["TouchIt", "CornerIt", "DoubleTapIt", "DragIt"]
SACCADE_GAMES  = ["CornerIt", "DoubleTapIt", "PinchIt", "SlideIt"]

# ── Sensor / sync ─────────────────────────────────────────────────────────────
TARGET_HZ        = 30
SYNC_TOLERANCE_MS = 34        # merge_asof window for gaze↔touch alignment
GAZE_SCALE       = 0.01       # raw gaze units → centimetres

# ── I-VT gaze algorithm ───────────────────────────────────────────────────────
VELOCITY_THRESHOLD_CM_S   = 20.0   # below = fixation, above = saccade
MIN_FIXATION_DURATION_SEC = 0.100  # 100 ms clinical standard

# ── Dragging stroke filter ────────────────────────────────────────────────────
MIN_STROKE_DURATION_SEC = 1.1
MIN_STROKE_FRAMES       = 35

# ── Age classification ────────────────────────────────────────────────────────
# Unified threshold — was inconsistent (50 / 60 / 65) across scripts 09, 11, 13
AGE_THRESHOLD_OLD = 60

# ── Feature filtering ─────────────────────────────────────────────────────────
MAX_INTER_FEATURE_CORR  = 0.85   # drop one of any pair above this
MIN_FEATURE_TARGET_CORR = 0.25   # drop features below this correlation with target
MISSING_DROP_THRESHOLD  = 0.30   # drop features missing in >30 % of rows

# ── ML defaults ───────────────────────────────────────────────────────────────
CV_FOLDS            = 5
XGB_N_ESTIMATORS    = 100
XGB_MAX_DEPTH       = 3
XGB_LEARNING_RATE   = 0.1
RF_N_ESTIMATORS     = 100
RANDOM_STATE        = 42
