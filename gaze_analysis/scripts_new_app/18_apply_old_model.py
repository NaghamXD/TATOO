"""
18_apply_old_model.py
=====================
Apply models trained on OLD-APP data to predict age (old vs young) for NEW-APP participants.

Steps:
  1. For each of the 6 shared games, train XGBoost on the old MASTER (which has labels).
  2. Align features to those that exist in both old and new data.
  3. Predict probabilities for new-app participants (NaN features handled by XGBoost natively).
  4. Combine game predictions into a weighted ensemble (weights = per-game AUC from old data).
  5. Save results table + visualisations.

Outputs → gaze_analysis/results/new_app/old_model_transfer/
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DIR as OLD_PROCESSED_DIR,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, RANDOM_STATE,
)
from config_new_app import (
    PROCESSED_DIR as NEW_PROCESSED_DIR,
    RESULTS_DIR,
)

OUT_DIR = RESULTS_DIR / "old_model_transfer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET   = "Target_Age_Old"
AGE_THRESHOLD = 60

# Only the 6 games present in both datasets
SHARED_GAMES = ["TouchIt", "CornerIt", "DoubleTapIt", "PinchIt", "SlideIt", "DragIt"]

# AUC weights from old-dataset script 14 (Touch+Gaze condition)
GAME_WEIGHTS = {
    "TouchIt":     0.838,
    "CornerIt":    0.961,
    "DoubleTapIt": 0.966,
    "PinchIt":     0.935,
    "SlideIt":     0.967,
    "DragIt":      0.955,
}

# Columns that are never ML features
NON_FEATURE_COLS = {
    "Patient_ID", "Age", "Target_Age_Old", "Target_Visual_Impaired", "Game",
    "Correct_Taps", "Outside_Taps",
    "Flight_Duration_sec", "Touch_Duration_sec",
    "Gaze_Pct_Within_Screen", "Gaze_Pct_Outside_Screen",
    "Pressure_Med_Pct",
}


def get_feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def train_on_old(game):
    """Train XGBoost on old-app MASTER file. Return fitted model + feature list."""
    fp = Path(str(OLD_PROCESSED_DIR)) / f"MASTER_{game}.csv"
    if not fp.exists():
        return None, None

    df = pd.read_csv(fp).dropna(subset=[TARGET])
    if len(df) < 8:
        print(f"  [{game}] old data too small ({len(df)} rows), skipping.")
        return None, None

    feature_cols = get_feature_cols(df)
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].astype(int)

    counts = y.value_counts()
    spw = counts.get(0, 1) / max(counts.get(1, 1), 1)

    model = xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X, y)
    return model, feature_cols


def predict_on_new(model, feature_cols, game):
    """Use trained model to predict probabilities for new-app participants."""
    fp = Path(str(NEW_PROCESSED_DIR)) / f"MASTER_{game}.csv"
    if not fp.exists():
        return pd.DataFrame()

    df = pd.read_csv(fp)
    df["Patient_ID"] = df["Patient_ID"].astype(str).str.strip()

    # Keep only features the model was trained on; fill missing with NaN (XGBoost handles it)
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  [{game}] {len(missing)} features not in new data (will be NaN): {missing[:4]}{'...' if len(missing)>4 else ''}")

    X_new = df.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce")
    probs = model.predict_proba(X_new)[:, 1]

    return pd.DataFrame({
        "Patient_ID": df["Patient_ID"].values,
        f"{game}_prob": probs,
        f"{game}_pred": (probs >= 0.5).astype(int),
    })


def weighted_ensemble(game_preds):
    """
    Combine per-game probabilities into one prediction per participant.
    Weight = game AUC (old dataset). Missing game contribution is ignored.
    """
    all_ids = set()
    for df in game_preds.values():
        all_ids.update(df["Patient_ID"].tolist())

    records = []
    for pid in sorted(all_ids):
        total_w, weighted_sum = 0.0, 0.0
        game_votes = {}
        for game, df in game_preds.items():
            row = df[df["Patient_ID"] == pid]
            if row.empty:
                continue
            prob = row[f"{game}_prob"].values[0]
            w    = GAME_WEIGHTS.get(game, 0.5)
            weighted_sum += w * prob
            total_w      += w
            game_votes[game] = prob

        if total_w == 0:
            continue

        final_prob = weighted_sum / total_w
        records.append({
            "Patient_ID":      pid,
            "Ensemble_Prob":   round(final_prob, 4),
            "Predicted_Class": int(final_prob >= 0.5),
            "Prediction":      "Old (≥60)" if final_prob >= 0.5 else "Young (<60)",
            "Confidence":      f"{max(final_prob, 1-final_prob)*100:.1f}%",
            "Games_Used":      len(game_votes),
            **{f"{g}_prob": round(game_votes.get(g, np.nan), 4) for g in SHARED_GAMES},
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_game_heatmap(ensemble_df):
    """Heatmap: participants × games, coloured by predicted probability."""
    prob_cols = [f"{g}_prob" for g in SHARED_GAMES if f"{g}_prob" in ensemble_df.columns]
    hmap = ensemble_df.set_index("Patient_ID")[prob_cols].copy()
    hmap.columns = [c.replace("_prob", "") for c in hmap.columns]

    fig, ax = plt.subplots(figsize=(10, max(4, len(hmap) * 0.5 + 2)))
    sns.heatmap(
        hmap.astype(float), annot=True, fmt=".2f",
        cmap="RdYlGn_r", vmin=0, vmax=1, center=0.5,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "P(Old)"},
        ax=ax,
    )
    ax.set_title("Per-Game Age Prediction Probabilities\n(Old model → New data | Red = Old, Green = Young)", pad=12)
    ax.set_xlabel("Game")
    ax.set_ylabel("Participant")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_game_probability_heatmap.png", dpi=150)
    plt.close(fig)
    print("[saved] per_game_probability_heatmap.png")


def plot_ensemble_bar(ensemble_df):
    """Horizontal bar chart of ensemble probability per participant."""
    df_sorted = ensemble_df.sort_values("Ensemble_Prob", ascending=True).copy()
    colors = ["#E8875A" if p >= 0.5 else "#5B8DB8" for p in df_sorted["Ensemble_Prob"]]

    fig, ax = plt.subplots(figsize=(8, max(4, len(df_sorted) * 0.5 + 2)))
    bars = ax.barh(df_sorted["Patient_ID"], df_sorted["Ensemble_Prob"], color=colors, alpha=0.88)
    ax.axvline(0.5, color="black", lw=1.2, ls="--", label="Decision boundary (0.5)")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Weighted Ensemble P(Old ≥ 60)")
    ax.set_title("Age Prediction — New App Participants\n(Trained on old-app data)")

    # Annotate with prediction label
    for bar, (_, row) in zip(bars, df_sorted.iterrows()):
        label = "Old" if row["Ensemble_Prob"] >= 0.5 else "Young"
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{label} ({row['Confidence']})", va="center", fontsize=9)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#E8875A", label="Predicted: Old (≥60)"),
        Patch(color="#5B8DB8", label="Predicted: Young (<60)"),
    ], loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ensemble_prediction_bar.png", dpi=150)
    plt.close(fig)
    print("[saved] ensemble_prediction_bar.png")


def plot_feature_importance_grid(game_models):
    """2×3 grid of top-5 feature importance per game (from old-data trained models)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, (game, (model, feat_cols)) in enumerate(game_models.items()):
        if model is None:
            axes[i].set_visible(False)
            continue
        imp = pd.DataFrame({"Feature": feat_cols, "Importance": model.feature_importances_})
        imp = imp.sort_values("Importance", ascending=False).head(5)
        axes[i].barh(imp["Feature"][::-1], imp["Importance"][::-1], color="#5FAD75", alpha=0.85)
        axes[i].set_title(f"{game}", fontweight="bold")
        axes[i].set_xlabel("Importance")
        axes[i].tick_params(labelsize=8)

    fig.suptitle("Top-5 Features per Game (trained on old-app data)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance_grid.png", dpi=150)
    plt.close(fig)
    print("[saved] feature_importance_grid.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("OLD MODEL → NEW DATA TRANSFER")
    print(f"Training games : {SHARED_GAMES}")
    print(f"Old data       : {OLD_PROCESSED_DIR}")
    print(f"New data       : {NEW_PROCESSED_DIR}")
    print("=" * 60)

    game_models = {}
    game_preds  = {}

    for game in SHARED_GAMES:
        print(f"\n── {game} ──")
        model, feat_cols = train_on_old(game)
        game_models[game] = (model, feat_cols)

        if model is None:
            continue

        print(f"  Trained on {sum(1 for _ in feat_cols)} features from old data.")
        pred_df = predict_on_new(model, feat_cols, game)

        if pred_df.empty:
            print(f"  No new-app data found for {game}.")
            continue

        game_preds[game] = pred_df
        old_count = pred_df[f"{game}_pred"].sum()
        print(f"  New-app predictions: {old_count}/{len(pred_df)} → Old, {len(pred_df)-old_count}/{len(pred_df)} → Young")

    if not game_preds:
        print("\nNo predictions generated.")
    else:
        # Ensemble
        ensemble_df = weighted_ensemble(game_preds)

        print("\n" + "=" * 60)
        print("FINAL ENSEMBLE PREDICTIONS")
        print("=" * 60)
        print(ensemble_df[["Patient_ID", "Prediction", "Ensemble_Prob", "Confidence", "Games_Used"]].to_string(index=False))

        old_n   = (ensemble_df["Predicted_Class"] == 1).sum()
        young_n = (ensemble_df["Predicted_Class"] == 0).sum()
        print(f"\nSummary: {old_n} predicted Old, {young_n} predicted Young (n={len(ensemble_df)})")

        # Save
        out_csv = OUT_DIR / "predictions.csv"
        ensemble_df.to_csv(out_csv, index=False)
        print(f"\n[saved] predictions.csv")

        # Plots
        plot_per_game_heatmap(ensemble_df)
        plot_ensemble_bar(ensemble_df)
        plot_feature_importance_grid(game_models)

        print(f"\nAll outputs → {OUT_DIR}")
