"""
Script 17 – Per-Patient SHAP Modality Attribution
==================================================
Explains WHAT drove each patient's age-group classification, at the
modality level: how much did Touch behavior, Gaze behavior, and Eye-Hand
Sync each contribute to that specific patient's prediction?

Uses Leave-One-Patient-Out CV (same leakage-safe scheme as script 15) on the
"Touch + Gaze" (full-feature) condition, and decomposes each held-out
patient's prediction into per-feature SHAP contributions via XGBoost's
native exact TreeSHAP (`pred_contribs=True`) — no external `shap` package
needed.

Outputs → gaze_analysis/results/per_patient_ablation/
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

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

# ── feature groups: same definitions as script 15 ────────────────────────────

TOUCH_FEATURES = [
    "First_Reaction_Time_sec",
    "Flight_Touch_Ratio",
    "Pressure_High_Pct",
    "Pressure_Low_Pct",
    "Game_Duration_sec",
    "Mean_Action_Duration_sec",
    "Spatial_Jerk_Magnitude",
    "Mean_Pressure_Overall",
    "Pressure_Jitter_Stationary",
    "Accuracy_Ratio",
    "Number_of_Taps",
]

GAZE_FEATURES = [
    "Gaze_In_Out_Ratio",
    "Fixation_Count",
    "Mean_Fixation_Duration_sec",
    "Mean_Saccadic_Amplitude_cm",
    "Peak_Saccadic_Velocity_cm_s",
    "Saccade_Frequency_Hz",
]

SYNC_FEATURES = [
    "Mean_Eye_Hand_Distance_cm",
    "Directional_Velocity_Alignment",
    "True_Eye_Hand_Latency_sec",
]

COMBINED_FEATURES = TOUCH_FEATURES + GAZE_FEATURES + SYNC_FEATURES

FEATURE_GROUP = {f: "Touch" for f in TOUCH_FEATURES}
FEATURE_GROUP.update({f: "Gaze" for f in GAZE_FEATURES})
FEATURE_GROUP.update({f: "Sync" for f in SYNC_FEATURES})

GROUPS = ["Touch", "Gaze", "Sync"]
COLORS = {"Touch": "#5B8DB8", "Gaze": "#E8875A", "Sync": "#9B6FB0"}


# ─────────────────────────────────────────────────────────────────────────────
# helpers (mirrors script 15)
# ─────────────────────────────────────────────────────────────────────────────

def available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Keep only columns that exist and have ≥50 % non-null values."""
    return [c for c in cols if c in df.columns and df[c].notna().mean() >= 0.50]


def make_xgb(df: pd.DataFrame) -> xgb.XGBClassifier:
    counts = df[TARGET].value_counts()
    spw = counts.get(0, 1) / max(counts.get(1, 1), 1)
    return xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# core LOPO + SHAP loop (Touch + Gaze condition only)
# ─────────────────────────────────────────────────────────────────────────────

def lopo_shap(df: pd.DataFrame, features: list[str]) -> list[dict]:
    """LOPO over the Combined condition; returns long-format SHAP records.

    Each held-out patient's prediction is decomposed into per-feature SHAP
    contributions via XGBoost's exact TreeSHAP (pred_contribs=True), computed
    on the SAME scaled feature space the model was trained/predicted on.
    """
    records = []
    if not features or df[TARGET].nunique() < 2:
        return records

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

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s  = scaler.transform(X_test)

        clf = make_xgb(train_df)
        clf.fit(X_train_s, train_df[TARGET])
        booster = clf.get_booster()

        dtest = xgb.DMatrix(X_test_s, feature_names=features)
        contribs = booster.predict(dtest, pred_contribs=True)   # (1, n_features + 1)
        margin   = booster.predict(dtest, output_margin=True)   # (1,)

        # additive property: sum(contribs incl. bias) == margin
        assert np.allclose(contribs[0].sum(), margin[0], atol=1e-4), \
            "SHAP contributions do not sum to model margin"

        prob = clf.predict_proba(X_test_s)[0, 1]
        pred = int(prob >= 0.5)
        patient_id = test_df["Patient_ID"].iloc[0]
        y_true = int(test_df[TARGET].iloc[0])
        raw_values = test_df[features].iloc[0]

        for j, feat in enumerate(features):
            records.append({
                "Patient_ID":    patient_id,
                "Game":          test_df["Game"].iloc[0] if "Game" in test_df.columns else None,
                "Feature":       feat,
                "Feature_Group": FEATURE_GROUP[feat],
                "SHAP_value":    float(contribs[0, j]),
                "Feature_Value": raw_values[feat],
                "y_true":        y_true,
                "y_pred":        pred,
                "correct":       int(pred == y_true),
            })

    return records


def run_analysis() -> pd.DataFrame:
    all_records = []
    for game in ALL_GAMES:
        path = PROCESSED_DIR / f"MASTER_{game}.csv"
        if not path.exists():
            print(f"[skip] {game} – MASTER file not found")
            continue

        df = pd.read_csv(path).dropna(subset=[TARGET]).reset_index(drop=True)
        if "Game" not in df.columns:
            df["Game"] = game

        feats = available(df, COMBINED_FEATURES)
        print(f"{game:<14} n_feat={len(feats):2d}  (LOPO folds={len(df)})")

        all_records.extend(lopo_shap(df, feats))

    return pd.DataFrame(all_records)


# ─────────────────────────────────────────────────────────────────────────────
# aggregation: modality contribution fingerprint per patient
# ─────────────────────────────────────────────────────────────────────────────

def build_modality_contribution(long_df: pd.DataFrame) -> pd.DataFrame:
    long_df = long_df.copy()
    long_df["abs_shap"] = long_df["SHAP_value"].abs()

    grouped = (
        long_df.groupby(["Patient_ID", "Feature_Group"])["abs_shap"]
        .sum()
        .unstack("Feature_Group")
        .reindex(columns=GROUPS, fill_value=0.0)
    )

    totals = grouped.sum(axis=1)
    pct = grouped.div(totals, axis=0) * 100
    pct.columns = [f"{g}_pct" for g in GROUPS]
    pct = pct.reset_index()

    pct["dominant_modality"] = pct[[f"{g}_pct" for g in GROUPS]].idxmax(axis=1).str.replace("_pct", "", regex=False)
    return pct


# ─────────────────────────────────────────────────────────────────────────────
# visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_stacked_bar(contrib: pd.DataFrame) -> None:
    ordered = contrib.sort_values(["dominant_modality", "Touch_pct"], ascending=[True, False])
    x = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(16, 6))
    bottom = np.zeros(len(ordered))
    for g in GROUPS:
        vals = ordered[f"{g}_pct"].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=g, color=COLORS[g], width=0.8)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(ordered["Patient_ID"], rotation=90, fontsize=7)
    ax.set_ylabel("Share of |SHAP| contribution (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        "Per-Patient Modality Contribution to Age-Group Classification\n"
        "(% of total |SHAP| from Touch+Gaze model attributable to each modality, "
        "grouped by dominant modality)"
    )
    ax.legend(title="Modality", loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "modality_contribution_stacked_bar.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PER-PATIENT SHAP MODALITY ATTRIBUTION (Touch + Gaze model, LOPO)")
    print(f"Target: {TARGET}")
    print("=" * 60)

    long_results = run_analysis()
    long_results.to_csv(OUT_DIR / "per_patient_shap_long.csv", index=False)
    print(f"\n[saved] per_patient_shap_long.csv  ({len(long_results)} rows)")

    contrib = build_modality_contribution(long_results)
    contrib.to_csv(OUT_DIR / "per_patient_modality_contribution.csv", index=False)
    print(f"[saved] per_patient_modality_contribution.csv  ({len(contrib)} patients)")

    plot_stacked_bar(contrib)
    print("[saved] modality_contribution_stacked_bar.png")

    print("\n" + "=" * 60)
    print("SUMMARY — dominant modality across patients")
    print("=" * 60)
    counts = contrib["dominant_modality"].value_counts()
    for g in GROUPS:
        print(f"  {g:<6} dominant: {int(counts.get(g, 0)):2d} patients")
    print(f"\n  Mean contribution shares: "
          f"Touch={contrib['Touch_pct'].mean():.1f}%  "
          f"Gaze={contrib['Gaze_pct'].mean():.1f}%  "
          f"Sync={contrib['Sync_pct'].mean():.1f}%")

    print(f"\nAll outputs → {OUT_DIR}")
