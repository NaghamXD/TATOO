"""
17_weighted_ensemble.py — New App  [PREPARED — NOT YET ACTIVE]

Weighted cross-game age prediction. Each game's XGBoost probability is
weighted by its ROC-AUC on the old dataset, giving higher influence to
games that historically discriminate age better.

Requires: Target_Age_Old labels attached to MASTER files (run attach_labels.py first).
Activate:  Uncomment 'new-app-ensemble' target in the Makefile.

Weights source: script 14 results on old dataset (gaze_added_value_results.csv)
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from pathlib import Path

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_new_app import (
    PROCESSED_DIR, RESULTS_DIR, ALL_GAMES,
    CV_FOLDS, XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, RANDOM_STATE,
)

processed_dir = str(PROCESSED_DIR)
results_dir   = str(RESULTS_DIR)
os.makedirs(results_dir, exist_ok=True)

TARGET_COLUMN = 'Target_Age_Old'

# AUC weights from old-dataset script 14 results (Touch Only condition).
# PinchItOut proxied from PinchIt; HoldIt proxied from mean of dragging games.
GAME_WEIGHTS = {
    'TouchIt':    0.744,
    'CornerIt':   0.870,
    'DoubleTapIt':0.962,
    'PinchIt':    0.860,
    'SlideIt':    0.967,
    'DragIt':     0.955,
    'PinchItOut': 0.860,   # proxy: same as PinchIt until measured
    'HoldIt':     0.890,   # proxy: mean of dragging games
}

COLS_TO_DROP = [
    'Patient_ID', 'Age', 'Target_Age_Old', 'Target_Visual_Impaired',
    'Correct_Taps', 'Outside_Taps',
    'Flight_Duration_sec', 'Touch_Duration_sec',
    'Gaze_Pct_Within_Screen', 'Gaze_Pct_Outside_Screen',
    'Pressure_Med_Pct', 'Game',
]


def train_per_game_models(game_list, cv_folds):
    """Train one XGBoost per game; return per-participant predicted probabilities."""
    all_probs = {}   # {patient_id: {game: prob}}

    for game in game_list:
        fp = os.path.join(processed_dir, f"MASTER_{game}.csv")
        if not os.path.exists(fp):
            continue

        df = pd.read_csv(fp).dropna(subset=[TARGET_COLUMN])
        if len(df) < cv_folds * 2:
            print(f"  {game}: skipped (n={len(df)} too small for CV).")
            continue

        X = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
        X = X.apply(pd.to_numeric, errors='coerce')
        y = df[TARGET_COLUMN]
        ids = df['Patient_ID'].astype(str).str.strip().values

        counts = y.value_counts()
        spw    = counts.get(0, 1) / max(counts.get(1, 1), 1)
        model  = xgb.XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE, scale_pos_weight=spw,
            eval_metric='logloss', random_state=RANDOM_STATE, verbosity=0,
        )

        cv   = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        probs = np.zeros(len(y))

        for train_idx, test_idx in cv.split(X, y):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            probs[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

        for pid, prob in zip(ids, probs):
            all_probs.setdefault(pid, {})[game] = prob

        print(f"  {game}: AUC={roc_auc_score(y, probs):.3f}")

    return all_probs


def weighted_ensemble_predict(all_probs, game_weights):
    """Combine per-game probabilities using AUC weights."""
    records = []
    for pid, game_probs in all_probs.items():
        total_weight = 0.0
        weighted_sum = 0.0
        for game, prob in game_probs.items():
            w = game_weights.get(game, 0.5)
            weighted_sum  += w * prob
            total_weight  += w
        final_prob = weighted_sum / total_weight if total_weight > 0 else 0.5
        records.append({'Patient_ID': pid, 'Ensemble_Prob': final_prob,
                        'Ensemble_Pred': int(final_prob >= 0.5),
                        'Games_Used': list(game_probs.keys())})
    return pd.DataFrame(records)


if __name__ == "__main__":
    print(f"\n{'='*60}\nWEIGHTED ENSEMBLE — New App\nTarget: {TARGET_COLUMN}\n{'='*60}")

    # Check labels exist
    has_labels = any(
        TARGET_COLUMN in pd.read_csv(os.path.join(processed_dir, f"MASTER_{g}.csv"), nrows=1).columns
        for g in ALL_GAMES
        if os.path.exists(os.path.join(processed_dir, f"MASTER_{g}.csv"))
    )
    if not has_labels:
        print("\nNo Target_Age_Old labels found. Run attach_labels.py first.")
    else:
        all_probs = train_per_game_models(ALL_GAMES, CV_FOLDS)
        ensemble_df = weighted_ensemble_predict(all_probs, GAME_WEIGHTS)

        # Load true labels from any MASTER file
        true_labels = {}
        for game in ALL_GAMES:
            fp = os.path.join(processed_dir, f"MASTER_{game}.csv")
            if os.path.exists(fp):
                df = pd.read_csv(fp)
                if TARGET_COLUMN in df.columns:
                    for _, row in df.iterrows():
                        true_labels[str(row['Patient_ID']).strip()] = int(row[TARGET_COLUMN])

        ensemble_df['True_Label'] = ensemble_df['Patient_ID'].map(true_labels)
        ensemble_df = ensemble_df.dropna(subset=['True_Label'])

        y_true = ensemble_df['True_Label'].astype(int)
        y_pred = ensemble_df['Ensemble_Pred']
        y_prob = ensemble_df['Ensemble_Prob']

        print(f"\nEnsemble results ({len(ensemble_df)} participants):")
        print(f"  Accuracy : {accuracy_score(y_true, y_pred)*100:.1f}%")
        print(f"  ROC-AUC  : {roc_auc_score(y_true, y_prob):.3f}")
        print(f"  F1       : {f1_score(y_true, y_pred, zero_division=0):.3f}")

        out_csv = os.path.join(results_dir, "Ensemble_Predictions.csv")
        ensemble_df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")

        # Probability distribution plot
        plt.figure(figsize=(8, 5))
        sns.histplot(data=ensemble_df, x='Ensemble_Prob', hue='True_Label',
                     bins=20, kde=True, palette={0: 'steelblue', 1: 'tomato'})
        plt.axvline(0.5, color='black', ls='--', lw=1)
        plt.title("Weighted Ensemble Probability Distribution\n(Blue=Young, Red=Old)")
        plt.xlabel("Predicted Probability of Age ≥ 60")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "Ensemble_Probability_Distribution.png"), dpi=300)
        plt.close()
