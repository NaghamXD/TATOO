"""
config_new_app.py — Configuration for the new-application pipeline.

Extends config.py (shared base constants) and overrides paths + game lists
to point at the new-app raw data. The old pipeline's config.py is untouched.
"""
from config import *          # inherit all shared constants
from pathlib import Path

ROOT = Path(__file__).parent

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR       = ROOT / "data" / "raw_new_app"
PROCESSED_DIR = ROOT / "data" / "processed_new_app"
RESULTS_DIR   = ROOT / "results" / "new_app"

# ── Games ─────────────────────────────────────────────────────────────────────
TAPPING_GAMES  = ["TouchIt", "CornerIt", "DoubleTapIt"]
DRAGGING_GAMES = ["PinchIt", "SlideIt", "DragIt", "PinchItOut"]
HOLD_GAMES     = ["HoldIt"]
ALL_GAMES      = TAPPING_GAMES + DRAGGING_GAMES + HOLD_GAMES

# Game → gaze feature assignment
FIXATION_GAMES = ["TouchIt", "CornerIt", "DoubleTapIt", "DragIt", "HoldIt"]
SACCADE_GAMES  = ["CornerIt", "DoubleTapIt", "PinchIt", "SlideIt", "PinchItOut"]
