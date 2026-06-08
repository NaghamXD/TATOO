"""
Script 16 – Per-Patient Accuracy Score Visualization
=====================================================
Turns script 15's leakage-free per-patient LOPO predictions into a per-patient
"how accurate was the model for THIS patient" score, broken down by feature
condition (Touch Only / Gaze Only / Touch + Gaze), and visualizes it as:
  - a heatmap of accuracy (annotated with confidence) across all 40 patients
  - boxplots showing the spread of accuracy/confidence scores per condition

Reads → gaze_analysis/results/per_patient_ablation/per_patient_lopo_results.csv
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
import seaborn as sns

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from config import RESULTS_DIR

OUT_DIR = RESULTS_DIR / "per_patient_ablation"
IN_PATH = OUT_DIR / "per_patient_lopo_results.csv"

CONDITIONS = ["Touch Only", "Gaze Only", "Touch + Gaze"]
COND_KEY   = {"Touch Only": "Touch", "Gaze Only": "Gaze", "Touch + Gaze": "Combined"}
COLORS     = {"Touch": "#5B8DB8", "Gaze": "#E8875A", "Combined": "#5FAD75"}


# ─────────────────────────────────────────────────────────────────────────────
# scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_scores(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient; columns = {Touch,Gaze,Combined} x {accuracy,confidence}."""
    df = long_df.copy()
    df["prob_true"] = np.where(df["y_true"] == 1, df["y_prob"], 1 - df["y_prob"])

    grouped = df.groupby(["Patient_ID", "Condition"]).agg(
        accuracy=("correct", "mean"),
        confidence=("prob_true", "mean"),
    )

    wide = grouped.unstack("Condition")
    wide.columns = [f"{COND_KEY[cond]}_{metric}" for metric, cond in wide.columns]
    wide = wide.reset_index()

    # ground-truth age group is constant per patient across games/conditions
    truth = df.groupby("Patient_ID")["y_true"].first().rename("Age_Group")
    wide = wide.merge(truth, on="Patient_ID")
    wide["Age_Group_Label"] = wide["Age_Group"].map({1: "Old (≥60)", 0: "Young (<60)"})

    cols = ["Patient_ID", "Age_Group", "Age_Group_Label"]
    for key in ["Touch", "Gaze", "Combined"]:
        cols += [f"{key}_accuracy", f"{key}_confidence"]
    return wide[cols]


# ─────────────────────────────────────────────────────────────────────────────
# heatmap: rows = patients, cols = conditions, color = accuracy, text = acc/conf
# ─────────────────────────────────────────────────────────────────────────────

AGE_GROUP_COLORS = {"Old (≥60)": "#C0504D", "Young (<60)": "#4F81BD"}


def plot_heatmap(scores: pd.DataFrame) -> None:
    ordered = scores.sort_values("Combined_accuracy", ascending=False)
    keys = ["Touch", "Gaze", "Combined"]
    labels = ["Touch Only", "Gaze Only", "Touch + Gaze"]

    acc  = ordered[[f"{k}_accuracy"   for k in keys]].to_numpy()
    conf = ordered[[f"{k}_confidence" for k in keys]].to_numpy()
    annot = np.array([[f"{a:.2f}\n({c:.2f})" for a, c in zip(arow, crow)]
                      for arow, crow in zip(acc, conf)])

    # y-tick labels carry the patient's ground-truth age group alongside their ID
    ytick_labels = [f"{pid}  —  {grp}"
                    for pid, grp in zip(ordered["Patient_ID"], ordered["Age_Group_Label"])]

    fig, ax = plt.subplots(figsize=(7, 14))
    sns.heatmap(
        acc, annot=annot, fmt="", cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        yticklabels=ytick_labels, xticklabels=labels,
        cbar_kws={"label": "Accuracy (text shows accuracy / confidence)"},
        ax=ax,
    )
    # color each y-tick label by the patient's true age group, so the
    # ground truth is readable at a glance alongside the prediction accuracy
    for tick in ax.get_yticklabels():
        group_label = tick.get_text().split("—")[-1].strip()
        tick.set_color(AGE_GROUP_COLORS.get(group_label, "black"))

    ax.set_title(
        "Per-Patient Accuracy by Feature Condition\n"
        "(color = accuracy across 6 games; annotation = accuracy / mean confidence in true class;\n"
        "row label color = ground-truth age group)\n"
        "sorted by Touch+Gaze accuracy",
        fontsize=11,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Patient  (true age group)")

    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=10, label=g)
               for g, c in AGE_GROUP_COLORS.items()]
    ax.legend(handles=handles, title="Ground truth", loc="upper left",
              bbox_to_anchor=(1.18, 1.0), frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_heatmap.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# distribution plots: spread of per-patient scores, one box per condition
# ─────────────────────────────────────────────────────────────────────────────

def plot_distribution(scores: pd.DataFrame, metric: str, label: str, fname: str) -> None:
    keys = ["Touch", "Gaze", "Combined"]
    long = scores.melt(
        id_vars="Patient_ID",
        value_vars=[f"{k}_{metric}" for k in keys],
        var_name="Condition", value_name=label,
    )
    long["Condition"] = long["Condition"].str.replace(f"_{metric}", "", regex=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=long, x="Condition", y=label, order=keys,
        palette=COLORS, showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="black", markeredgecolor="black", markersize=7),
        ax=ax,
    )
    sns.stripplot(
        data=long, x="Condition", y=label, order=keys,
        color="black", alpha=0.4, size=4, jitter=0.15, ax=ax,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("")
    ax.set_title(f"Per-Patient {label} Spread by Feature Condition\n(40 patients, ◆ = mean)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not IN_PATH.exists():
        sys.exit(f"[error] {IN_PATH} not found — run script 15 first.")

    long_df = pd.read_csv(IN_PATH)
    scores = compute_scores(long_df)
    scores.to_csv(OUT_DIR / "per_patient_accuracy_scores.csv", index=False)
    print(f"[saved] per_patient_accuracy_scores.csv  ({len(scores)} patients)")

    print("\nMean accuracy across patients:")
    for key in ["Touch", "Gaze", "Combined"]:
        print(f"  {key:<9} acc={scores[f'{key}_accuracy'].mean():.3f}   "
              f"confidence={scores[f'{key}_confidence'].mean():.3f}")

    plot_heatmap(scores)
    print("[saved] accuracy_heatmap.png")

    plot_distribution(scores, "accuracy",   "Accuracy",   "accuracy_distribution_by_condition.png")
    plot_distribution(scores, "confidence", "Confidence", "confidence_distribution_by_condition.png")
    print("[saved] accuracy_distribution_by_condition.png")
    print("[saved] confidence_distribution_by_condition.png")

    print(f"\nAll outputs → {OUT_DIR}")
