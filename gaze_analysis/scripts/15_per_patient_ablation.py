"""
Script 15 – Per-Patient Ablation Analysis (Leave-One-Patient-Out)
==================================================================
Extends script 14's feature-set ablation (Touch Only / Gaze Only / Touch+Gaze)
from population-level mean metrics down to per-patient effects.

For every patient and game, each condition's prediction comes from a model
trained on the OTHER 39 patients (Leave-One-Patient-Out CV) — no model ever
scores a patient it was trained on, so per-patient deltas are leakage-free.

Outputs → gaze_analysis/results/per_patient_ablation/
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBClassifier

# ── resolve imports whether run from repo root or scripts/ ────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from config import (
    ALL_GAMES,
    PROCESSED_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
)

OUT_DIR = RESULTS_DIR / "per_patient_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Target_Age_Old"

# ── feature groups: same definitions as script 14 (ML model features only) ───

TOUCH_FEATURES = [
    "First_Reaction_Time_sec",
    "Flight_Touch_Ratio",
    "Pressure_High_Pct",
    "Pressure_Low_Pct",
    "Game_Duration_sec",
    "Mean_Action_Duration_sec",   # dragging games only
    "Spatial_Jerk_Magnitude",     # dragging games only
    "Mean_Pressure_Overall",      # tapping games only
    "Pressure_Jitter_Stationary", # tapping games only
    "Accuracy_Ratio",             # CornerIt / DoubleTapIt
    "Number_of_Taps",             # TouchIt only
]

GAZE_FEATURES = [
    "Gaze_In_Out_Ratio",
    "Fixation_Count",
    "Mean_Fixation_Duration_sec",
    "Mean_Saccadic_Amplitude_cm",
    "Peak_Saccadic_Velocity_cm_s",
    "Saccade_Frequency_Hz",
]

SYNC_FEATURES = [                  # require both gaze + touch
    "Mean_Eye_Hand_Distance_cm",
    "Directional_Velocity_Alignment",
    "True_Eye_Hand_Latency_sec",
]

CONDITIONS = {
    "Touch Only":   TOUCH_FEATURES,
    "Gaze Only":    GAZE_FEATURES,
    "Touch + Gaze": TOUCH_FEATURES + GAZE_FEATURES + SYNC_FEATURES,
}

COND_KEY = {
    "Touch Only":   "touch",
    "Gaze Only":    "gaze",
    "Touch + Gaze": "combined",
}


# ─────────────────────────────────────────────────────────────────────────────
# helpers (mirrors script 14)
# ─────────────────────────────────────────────────────────────────────────────

def available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Keep only columns that exist and have ≥50 % non-null values."""
    return [c for c in cols if c in df.columns and df[c].notna().mean() >= 0.50]


def make_xgb(df: pd.DataFrame) -> XGBClassifier:
    counts = df[TARGET].value_counts()
    spw = counts.get(0, 1) / max(counts.get(1, 1), 1)
    return XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )


def lopo_predict(df: pd.DataFrame, features: list[str]) -> pd.Series:
    """Leave-One-Patient-Out probabilities: each patient scored by a model
    trained only on the other patients (median imputed on the train fold)."""
    probs = pd.Series(np.nan, index=df.index, dtype=float)
    if not features or df[TARGET].nunique() < 2:
        return probs

    logo = LeaveOneGroupOut()
    groups = df["Patient_ID"].values
    X_full = df[features]
    y_full = df[TARGET]

    for train_idx, test_idx in logo.split(X_full, y_full, groups):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        assert not set(train_df["Patient_ID"]) & set(test_df["Patient_ID"]), \
            "leakage: held-out patient present in training fold"

        median = X_full.iloc[train_idx].median()
        X_train = X_full.iloc[train_idx].fillna(median)
        X_test  = X_full.iloc[test_idx].fillna(median)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    make_xgb(train_df)),
        ])
        pipe.fit(X_train, train_df[TARGET])
        probs.iloc[test_idx] = pipe.predict_proba(X_test)[:, 1]

    return probs


# ─────────────────────────────────────────────────────────────────────────────
# main analysis loop
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis() -> pd.DataFrame:
    records = []
    for game in ALL_GAMES:
        path = PROCESSED_DIR / f"MASTER_{game}.csv"
        if not path.exists():
            print(f"[skip] {game} – MASTER file not found")
            continue

        df = pd.read_csv(path).dropna(subset=[TARGET]).reset_index(drop=True)
        print(f"\n{'─'*60}")
        print(f"{game}  (n={len(df)} patients, LOPO folds={len(df)})")

        for cond, cols in CONDITIONS.items():
            feats = available(df, cols)
            probs = lopo_predict(df, feats)
            preds = (probs >= 0.5).astype(int)
            correct = (preds == df[TARGET]).astype(int)

            for i in range(len(df)):
                records.append({
                    "Patient_ID": df.loc[i, "Patient_ID"],
                    "Game":       game,
                    "Condition":  cond,
                    "y_true":     int(df.loc[i, TARGET]),
                    "y_prob":     probs.iloc[i],
                    "y_pred":     int(preds.iloc[i]) if not np.isnan(probs.iloc[i]) else np.nan,
                    "correct":    int(correct.iloc[i]) if not np.isnan(probs.iloc[i]) else np.nan,
                })

            valid = probs.notna()
            acc = correct[valid].mean() if valid.any() else np.nan
            print(f"  {cond:<16} n_feat={len(feats):2d}  LOPO Acc={acc:.3f}")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# per-patient deltas + flip categories
# ─────────────────────────────────────────────────────────────────────────────

def classify_flip(correct_before: float, correct_after: float) -> str:
    if np.isnan(correct_before) or np.isnan(correct_after):
        return "n/a"
    if correct_before == correct_after:
        return "no change"
    return "incorrect→correct" if correct_after > correct_before else "correct→incorrect"


def build_deltas(long_df: pd.DataFrame) -> pd.DataFrame:
    wide = long_df.pivot_table(
        index=["Patient_ID", "Game", "y_true"],
        columns="Condition",
        values=["y_prob", "correct"],
    )
    wide.columns = [f"{val}_{COND_KEY[cond]}" for val, cond in wide.columns]
    wide = wide.reset_index()

    wide["delta_gaze_added"] = wide["y_prob_combined"] - wide["y_prob_touch"]
    wide["delta_gaze_alone"] = wide["y_prob_gaze"]     - wide["y_prob_touch"]

    wide["flip_added"] = [
        classify_flip(b, a) for b, a in zip(wide["correct_touch"], wide["correct_combined"])
    ]
    wide["flip_alone"] = [
        classify_flip(b, a) for b, a in zip(wide["correct_touch"], wide["correct_gaze"])
    ]

    cols = [
        "Patient_ID", "Game", "y_true",
        "y_prob_touch", "y_prob_gaze", "y_prob_combined",
        "correct_touch", "correct_gaze", "correct_combined",
        "delta_gaze_added", "delta_gaze_alone",
        "flip_added", "flip_alone",
    ]
    return wide[cols]


def build_flip_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for game, sub in deltas.groupby("Game"):
        for flip_col, label in [("flip_added", "gaze_added"), ("flip_alone", "gaze_alone")]:
            counts = sub[flip_col].value_counts()
            rows.append({
                "Game": game,
                "Comparison": label,
                "incorrect→correct": int(counts.get("incorrect→correct", 0)),
                "correct→incorrect": int(counts.get("correct→incorrect", 0)),
                "no change":         int(counts.get("no change", 0)),
            })
    return pd.DataFrame(rows)


def print_summary(deltas: pd.DataFrame, flips: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY — per-patient effect of adding gaze (Touch+Gaze vs Touch)")
    print("=" * 60)
    total = flips[flips["Comparison"] == "gaze_added"][
        ["incorrect→correct", "correct→incorrect", "no change"]
    ].sum()
    print(
        f"  Across all games: "
        f"{total['incorrect→correct']} patient-games improved, "
        f"{total['correct→incorrect']} got worse, "
        f"{total['no change']} unchanged"
    )
    print(f"  Mean Δ probability (Touch+Gaze − Touch): {deltas['delta_gaze_added'].mean():+.3f}")

    print("\nSUMMARY — per-patient effect of gaze alone (Gaze vs Touch)")
    total = flips[flips["Comparison"] == "gaze_alone"][
        ["incorrect→correct", "correct→incorrect", "no change"]
    ].sum()
    print(
        f"  Across all games: "
        f"{total['incorrect→correct']} patient-games improved, "
        f"{total['correct→incorrect']} got worse, "
        f"{total['no change']} unchanged"
    )
    print(f"  Mean Δ probability (Gaze − Touch): {deltas['delta_gaze_alone'].mean():+.3f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PER-PATIENT ABLATION ANALYSIS (Leave-One-Patient-Out)")
    print(f"Target: {TARGET}")
    print("CV: LeaveOneGroupOut grouped by Patient_ID")
    print("=" * 60)

    long_results = run_analysis()
    long_results.to_csv(OUT_DIR / "per_patient_lopo_results.csv", index=False)
    print(f"\n[saved] per_patient_lopo_results.csv  ({len(long_results)} rows)")

    deltas = build_deltas(long_results)
    deltas.to_csv(OUT_DIR / "per_patient_deltas.csv", index=False)
    print(f"[saved] per_patient_deltas.csv  ({len(deltas)} rows)")

    flips = build_flip_summary(deltas)
    flips.to_csv(OUT_DIR / "flip_summary.csv", index=False)
    print(f"[saved] flip_summary.csv")

    print_summary(deltas, flips)
    print(f"All outputs → {OUT_DIR}")
