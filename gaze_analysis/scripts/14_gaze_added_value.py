"""
Script 14 – Gaze Added-Value Analysis
======================================
Ablation study answering two questions:
  Q1. Gaze-Only  vs Touch-Only  — can gaze alone match/beat touch alone?
  Q2. Touch+Gaze vs Touch-Only  — does adding gaze improve a touch baseline?

Only features marked "Used by the ML Model = Yes" in the feature dictionary
are included. Features marked No (Flight_Duration_sec, Touch_Duration_sec,
Pressure_Med_Pct, Correct_Taps, Outside_Taps, Gaze_Pct_Within_Screen,
Gaze_Pct_Outside_Screen) are intentionally excluded.

Outputs → gaze_analysis/results/gaze_added_value/
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
import seaborn as sns
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── resolve imports whether run from repo root or scripts/ ────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from config import (
    ALL_GAMES,
    CV_FOLDS,
    PROCESSED_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
)

OUT_DIR = RESULTS_DIR / "gaze_added_value"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Target_Age_Old"

# ── feature groups: ML model features only (Used by ML Model = Yes) ───────────

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

COLORS = {
    "Touch Only":   "#5B8DB8",
    "Gaze Only":    "#E8875A",
    "Touch + Gaze": "#5FAD75",
}


# ─────────────────────────────────────────────────────────────────────────────
# helpers
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


def run_cv(df: pd.DataFrame, features: list[str]) -> dict:
    """5×10 repeated stratified CV → mean ± std of Accuracy, ROC-AUC, F1."""
    if not features or df[TARGET].nunique() < 2:
        nan = dict(accuracy=np.nan, roc_auc=np.nan, f1=np.nan,
                   accuracy_std=np.nan, roc_auc_std=np.nan, f1_std=np.nan,
                   n_features=0)
        return nan

    X = df[features].fillna(df[features].median())
    y = df[TARGET]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    make_xgb(df)),
    ])
    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=10,
                                  random_state=RANDOM_STATE)
    scorers = {
        "accuracy": "accuracy",
        "roc_auc":  "roc_auc",
        "f1":       make_scorer(f1_score, zero_division=0),
    }
    res = cross_validate(pipe, X, y, cv=cv, scoring=scorers, n_jobs=-1)

    return dict(
        accuracy    =res["test_accuracy"].mean(),
        roc_auc     =res["test_roc_auc"].mean(),
        f1          =res["test_f1"].mean(),
        accuracy_std=res["test_accuracy"].std(),
        roc_auc_std =res["test_roc_auc"].std(),
        f1_std      =res["test_f1"].std(),
        n_features  =len(features),
    )


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

        df = pd.read_csv(path).dropna(subset=[TARGET])
        print(f"\n{'─'*60}")
        print(f"{game}  (n={len(df)},  old={df[TARGET].sum()},  young={len(df)-df[TARGET].sum()})")

        for cond, cols in CONDITIONS.items():
            feats = available(df, cols)
            m = run_cv(df, feats)
            print(
                f"  {cond:<16} "
                f"n_feat={m['n_features']:2d}  "
                f"AUC={m['roc_auc']:.3f}±{m['roc_auc_std']:.3f}  "
                f"Acc={m['accuracy']:.3f}  "
                f"F1={m['f1']:.3f}"
            )
            records.append({"Game": game, "Condition": cond, **m})

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY — mean ± std across all games")
    print("=" * 60)
    for cond in CONDITIONS:
        sub = df[df["Condition"] == cond]
        print(
            f"  {cond:<16}  "
            f"AUC={sub['roc_auc'].mean():.3f}±{sub['roc_auc'].std():.3f}  "
            f"Acc={sub['accuracy'].mean():.3f}  "
            f"F1={sub['f1'].mean():.3f}"
        )

    baseline = df[df["Condition"] == "Touch Only"].set_index("Game")
    print("\nADDED VALUE vs Touch-Only (mean Δ across games):")
    for cond in ["Gaze Only", "Touch + Gaze"]:
        sub = df[df["Condition"] == cond].set_index("Game")
        games_both = baseline.index.intersection(sub.index)
        d_auc = (sub.loc[games_both, "roc_auc"] - baseline.loc[games_both, "roc_auc"]).mean()
        d_acc = (sub.loc[games_both, "accuracy"] - baseline.loc[games_both, "accuracy"]).mean()
        d_f1  = (sub.loc[games_both, "f1"] - baseline.loc[games_both, "f1"]).mean()
        print(f"  {cond:<16}  ΔAUC={d_auc:+.3f}  ΔAcc={d_acc:+.3f}  ΔF1={d_f1:+.3f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_bars(df: pd.DataFrame, metric: str, label: str) -> None:
    games = [g for g in ALL_GAMES if g in df["Game"].values]
    conds = list(CONDITIONS.keys())
    x = np.arange(len(games))
    w = 0.26

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, cond in enumerate(conds):
        sub = df[df["Condition"] == cond].set_index("Game")
        vals = [sub.loc[g, metric]          if g in sub.index else np.nan for g in games]
        errs = [sub.loc[g, f"{metric}_std"] if g in sub.index else np.nan for g in games]
        ax.bar(x + (i - 1) * w, vals, w,
               label=cond, color=COLORS[cond], alpha=0.88,
               yerr=errs, capsize=3,
               error_kw=dict(elinewidth=1.2, ecolor="#555"))

    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=15, ha="right")
    ax.set_ylabel(label)
    ax.set_ylim(0, 1.08)
    ax.axhline(0.5, color="red", lw=0.8, ls="--", label="Chance (0.50)", alpha=0.6)
    ax.set_title(
        f"{label} by Feature Set and Game\n"
        f"(XGBoost · 5×10 Repeated Stratified CV · n=40 per game)"
    )
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"bar_{metric}.png", dpi=150)
    plt.close(fig)


def plot_delta_heatmap(df: pd.DataFrame, metric: str, label: str) -> None:
    games = [g for g in ALL_GAMES if g in df["Game"].values]
    baseline = df[df["Condition"] == "Touch Only"].set_index("Game")[metric]

    rows = {}
    for cond in ["Gaze Only", "Touch + Gaze"]:
        sub = df[df["Condition"] == cond].set_index("Game")[metric]
        rows[cond] = {g: sub.get(g, np.nan) - baseline.get(g, np.nan) for g in games}

    hmap = pd.DataFrame(rows, index=games)
    vmax = max(abs(hmap.values[~np.isnan(hmap.values)]).max(), 0.01)

    fig, ax = plt.subplots(figsize=(5, 6))
    sns.heatmap(
        hmap, annot=True, fmt=".3f",
        center=0, vmin=-vmax, vmax=vmax,
        cmap="RdYlGn", linewidths=0.5, linecolor="white",
        cbar_kws={"label": f"Δ{label} vs Touch Only"},
        ax=ax,
    )
    ax.set_title(f"Added Value of Gaze\n(Δ{label} vs Touch-Only baseline)")
    ax.set_xlabel("")
    ax.set_ylabel("Game")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"delta_heatmap_{metric}.png", dpi=150)
    plt.close(fig)


def plot_radar(df: pd.DataFrame) -> None:
    metrics = ["accuracy", "roc_auc", "f1"]
    labels  = ["Accuracy", "ROC-AUC", "F1"]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    for cond in CONDITIONS:
        sub = df[df["Condition"] == cond]
        vals = [sub[m].mean() for m in metrics] + [sub["accuracy"].mean()]
        ax.plot(angles, vals, "o-", lw=2, color=COLORS[cond], label=cond)
        ax.fill(angles, vals, alpha=0.10, color=COLORS[cond])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=8)
    ax.set_title("Mean Performance Across All Games", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.18))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "radar_summary.png", dpi=150)
    plt.close(fig)


def plot_feature_counts(df: pd.DataFrame) -> None:
    games = [g for g in ALL_GAMES if g in df["Game"].values]
    conds = list(CONDITIONS.keys())
    x = np.arange(len(games))
    w = 0.26

    fig, ax = plt.subplots(figsize=(11, 4))
    for i, cond in enumerate(conds):
        sub = df[df["Condition"] == cond].set_index("Game")
        vals = [int(sub.loc[g, "n_features"]) if g in sub.index else 0 for g in games]
        bars = ax.bar(x + (i - 1) * w, vals, w,
                      label=cond, color=COLORS[cond], alpha=0.88)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                        str(v), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=15, ha="right")
    ax.set_ylabel("Features used")
    ax.set_title("Feature Count per Condition and Game (after availability filter)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_counts.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GAZE ADDED-VALUE ANALYSIS")
    print(f"Target: {TARGET}")
    print("CV: RepeatedStratifiedKFold(5 splits × 10 repeats)")
    print("=" * 60)

    results = run_analysis()
    results.to_csv(OUT_DIR / "gaze_added_value_results.csv", index=False)
    print(f"\n[saved] gaze_added_value_results.csv")

    print_summary(results)

    for metric, label in [
        ("roc_auc",  "ROC-AUC"),
        ("accuracy", "Accuracy"),
        ("f1",       "F1 Score"),
    ]:
        plot_metric_bars(results, metric, label)
        plot_delta_heatmap(results, metric, label)
        print(f"[saved] bar_{metric}.png  delta_heatmap_{metric}.png")

    plot_radar(results)
    print("[saved] radar_summary.png")

    plot_feature_counts(results)
    print("[saved] feature_counts.png")

    print(f"\nAll outputs → {OUT_DIR}")
