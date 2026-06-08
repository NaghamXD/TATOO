import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress minor warnings for cleaner terminal output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
PROCESSED_DIR = "gaze_analysis/data/processed/"
RESULTS_DIR = "gaze_analysis/results/"
FEATURE_IMPORTANCE_DIR = os.path.join(RESULTS_DIR, "feature_importance")

# Toggle this between 'Target_Age_Old' and 'Target_Visual_Impaired'
TARGET_COLUMN = 'Target_Age_Old'

if not os.path.exists(FEATURE_IMPORTANCE_DIR):
    os.makedirs(FEATURE_IMPORTANCE_DIR)

def train_and_evaluate(game_name):
    file_path = os.path.join(PROCESSED_DIR, f"MASTER_{game_name}.csv")
    if not os.path.exists(file_path):
        return
        
    df = pd.read_csv(file_path)
    
    # 1. Clean Data: Drop patients missing the target label
    df = df.dropna(subset=[TARGET_COLUMN])
    
    if df.empty or len(df) < 10:
        print(f"⚠️ Not enough data to train {game_name}.")
        return

    # 2. Separate Features (X) and Target (y)
    cols_to_drop = [
        'Patient_ID', 'Age', 'Target_Age_Old', 'Target_Visual_Impaired',
        'Correct_Taps', 'Outside_Taps', 
        'Flight_Duration_sec', 'Touch_Duration_sec',
        'Gaze_Pct_Within_Screen', 'Gaze_Pct_Outside_Screen',
        'Pressure_Med_Pct'
    ]
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    X = X.apply(pd.to_numeric, errors='coerce')
    y = df[TARGET_COLUMN].astype(int)

    # --- 3. DYNAMIC CLASS IMBALANCE HANDLER ---
    # Count the classes (e.g., 25 Class 0, 15 Class 1)
    class_0_count = len(y[y == 0])
    class_1_count = len(y[y == 1])
    
    # scale_pos_weight = count(negative examples) / count(positive examples)
    # If Visual Impaired (25/15), weight is 1.67. If Age (21/18), weight is 1.16.
    dynamic_weight = class_0_count / class_1_count if class_1_count > 0 else 1.0

    # 4. Initialize the Clinical XGBoost Model
    # We use shallow trees (max_depth=2) to prevent overfitting on 40 patients
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=2,        
        learning_rate=0.05,
        subsample=0.8,      # Randomly sample 80% of data per tree to add robustness
        scale_pos_weight=dynamic_weight, # <--- The Magic Bullet for Imbalanced Data
        eval_metric='logloss',
        random_state=42
    )

    # 5. Cross-Validation (5-Fold Stratified)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # We score on multiple metrics, not just accuracy!
    scoring_metrics = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring_metrics)

    acc = cv_results['test_accuracy'].mean()
    auc = cv_results['test_roc_auc'].mean()
    prec = cv_results['test_precision'].mean()
    rec = cv_results['test_recall'].mean()
    f1 = cv_results['test_f1'].mean()

    print(f"⚖️  Class Balance: {class_0_count} Negative (0) | {class_1_count} Positive (1)")
    print(f"✅ Accuracy:  {acc*100:.1f}%")
    print(f"📊 ROC-AUC:   {auc:.3f}  (Ability to distinguish classes)")
    print(f"🎯 Precision: {prec:.3f}  (When it says they are positive, are they?)")
    print(f"🔍 Recall:    {rec:.3f}  (Out of all true positives, how many did it catch?)")
    print(f"🥇 F1-Score:  {f1:.3f}  (Harmonic mean of Precision and Recall)\n")

    # 6. Feature Importance (Train on all data to extract insights)
    model.fit(X, y)
    
    importances = model.feature_importances_
    feature_names = X.columns
    
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    # Drop features that XGBoost found completely useless (Importance == 0)
    importance_df = importance_df[importance_df['Importance'] > 0]
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)

    # If the model couldn't find any patterns, skip plotting
    if importance_df.empty:
        print(f"⚠️ XGBoost found no predictive features for {game_name}.")
        return

    # 7. Generate the Biomarker Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title(f"Top Digital Biomarkers for {game_name}\n(Predicting {TARGET_COLUMN})")
    plt.xlabel("XGBoost Feature Importance (Gain)")
    plt.ylabel("")
    plt.tight_layout()
    
    plot_path = os.path.join(FEATURE_IMPORTANCE_DIR, f"{game_name}_{TARGET_COLUMN}_Importance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    TARGET_GAMES = ['TouchIt', 'CornerIt', 'DoubleTapIt', 'PinchIt', 'SlideIt', 'DragIt']
    
    print(f"\n{'='*60}")
    print(f"🧠 TRAINING XGBOOST CLINICAL DIAGNOSTIC MODEL")
    print(f"🎯 Target: {TARGET_COLUMN}")
    print(f"{'='*60}\n")
    
    for game in TARGET_GAMES:
        print(f"🚀 Processing Game: {game}")
        train_and_evaluate(game)
        print("-" * 60)