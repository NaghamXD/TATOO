"""
11_train_model.py — New App  [PREPARED — NOT YET ACTIVE]

This script is ready to run but requires demographics (Target_Age_Old labels)
to be attached to the MASTER files first.

To activate:
  1. Run attach_labels.py once demographics are collected
  2. Uncomment the 'new-app-model' target in the Makefile
"""
import os
import sys
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
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

COLS_TO_DROP = [
    'Patient_ID', 'Age', 'Target_Age_Old', 'Target_Visual_Impaired',
    'Correct_Taps', 'Outside_Taps',
    'Flight_Duration_sec', 'Touch_Duration_sec',
    'Gaze_Pct_Within_Screen', 'Gaze_Pct_Outside_Screen',
    'Pressure_Med_Pct', 'Game',
]


def train_and_evaluate(game_name):
    fp = os.path.join(processed_dir, f"MASTER_{game_name}.csv")
    if not os.path.exists(fp):
        return None

    df = pd.read_csv(fp).dropna(subset=[TARGET_COLUMN])
    if len(df) < 10:
        print(f"  {game_name}: not enough labelled data ({len(df)} rows).")
        return None

    X = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
    X = X.apply(pd.to_numeric, errors='coerce')
    y = df[TARGET_COLUMN]

    counts = y.value_counts()
    spw = counts.get(0, 1) / max(counts.get(1, 1), 1)

    model = xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        scale_pos_weight=spw,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

    print(f"  {game_name}: Acc={acc.mean()*100:.1f}% AUC={auc.mean():.3f}")

    model.fit(X, y)
    imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    imp_df = imp_df.sort_values('Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis')
    plt.title(f"Top 10 Biomarkers — {game_name}\n({TARGET_COLUMN})")
    plt.tight_layout()
    out = os.path.join(results_dir, f"{game_name}_{TARGET_COLUMN}_Importance.png")
    plt.savefig(out, dpi=300)
    plt.close()

    return {'game': game_name, 'accuracy': acc.mean(), 'roc_auc': auc.mean()}


if __name__ == "__main__":
    print(f"\n{'='*60}\nXGBOOST MODEL TRAINING — New App\nTarget: {TARGET_COLUMN}\n{'='*60}")

    # Check that at least one MASTER file has labels
    any_labels = False
    for game in ALL_GAMES:
        fp = os.path.join(processed_dir, f"MASTER_{game}.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp, nrows=5)
            if TARGET_COLUMN in df.columns:
                any_labels = True
                break

    if not any_labels:
        print("\nNo Target_Age_Old column found in MASTER files.")
        print("Run attach_labels.py after demographics are collected, then re-run this script.")
    else:
        results = [train_and_evaluate(g) for g in ALL_GAMES]
        results = [r for r in results if r]
        if results:
            summary = pd.DataFrame(results)
            print(f"\n{summary.to_string(index=False)}")
