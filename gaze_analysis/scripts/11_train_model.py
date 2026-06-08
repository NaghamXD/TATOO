import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress harmless SciPy ties/variance warnings for clean console output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
PROCESSED_DIR = "gaze_analysis/data/processed/"
RESULTS_DIR = "gaze_analysis/results/"
FEATURE_IMPORTANCE_DIR = os.path.join(RESULTS_DIR, "feature_importance")
if not os.path.exists(FEATURE_IMPORTANCE_DIR):
    os.makedirs(FEATURE_IMPORTANCE_DIR)

# Change this to 'Target_Visual_Impaired' to run your second experiment!
# 'Target_Age_Old' is the label for whether a patient is 65 or older (1) or younger than 65 (0)
TARGET_COLUMN = 'Target_Age_Old'  

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def train_and_evaluate(game_name):
    file_path = os.path.join(PROCESSED_DIR, f"MASTER_{game_name}.csv")
    if not os.path.exists(file_path):
        return
        
    df = pd.read_csv(file_path)
    
    # 1. Clean Data: Drop patients who are missing the target demographic label
    df = df.dropna(subset=[TARGET_COLUMN])
    
    if df.empty or len(df) < 10:
        print(f"⚠️ Not enough labeled data to train {game_name}.")
        return

    # 2. Separate the Features (X) from the Target Answer (y)
    # We must hide the ID and Demographics from the AI so it doesn't cheat
    cols_to_drop = [
        # Demographics
        'Patient_ID', 'Age', 'Target_Age_Old', 'Target_Visual_Impaired',
        
        # Tapping Game Redundancies (TouchIt, CornerIt, DoubleTapIt)
        'Correct_Taps', 'Outside_Taps', 
        
        # Universal Redundancies
        'Flight_Duration_sec', 'Touch_Duration_sec',
        'Gaze_Pct_Within_Screen', 'Gaze_Pct_Outside_Screen',
        
        # Pressure Baseline
        'Pressure_Med_Pct'

        #gamename
        'Game'
    ]
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    y = df[TARGET_COLUMN]

    # Ensure all features are numeric (Convert any accidental strings)
    X = X.apply(pd.to_numeric, errors='coerce')

    # 3. Initialize the Clinical XGBoost Model
    # XGBoost handles NaNs automatically, which is perfect for sensor data
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,        # Kept shallow to prevent overfitting on small clinical datasets
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42
    )

    # 4. Cross-Validation (5-Fold Stratified)
    # We test the model on 5 different slices of your patients to guarantee it isn't memorizing data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    accuracies = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

    print(f"✅ Accuracy: {accuracies.mean()*100:.1f}% (± {accuracies.std()*100:.1f}%)")
    print(f"📊 ROC-AUC:  {auc_scores.mean():.3f}")

    # 5. Feature Importance (Train on all data to see what the AI learned)
    model.fit(X, y)
    
    # Extract the weight of every biomarker
    importances = model.feature_importances_
    feature_names = X.columns
    
    # Sort them from most important to least important
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10) # Top 10 only

    # 6. Generate the Biomarker Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title(f"Top 10 Digital Biomarkers for {game_name}\n(Predicting {TARGET_COLUMN})")
    plt.xlabel("XGBoost Feature Importance (Gain)")
    plt.ylabel("")
    plt.tight_layout()
    
    # Save the plot directly to your results folder
    plot_path = os.path.join(FEATURE_IMPORTANCE_DIR, f"{game_name}_{TARGET_COLUMN}_Importance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"🖼️  Saved Feature Importance chart to: {plot_path}\n")

if __name__ == "__main__":
    TARGET_GAMES = ['TouchIt', 'CornerIt', 'DoubleTapIt', 'PinchIt', 'SlideIt', 'DragIt']
    
    print(f"\n{'='*60}")
    print(f"🧠 TRAINING XGBOOST CLINICAL DIAGNOSTIC MODEL")
    print(f"🎯 Target: {TARGET_COLUMN}")
    print(f"{'='*60}\n")
    
    for game in TARGET_GAMES:
        print(f"🚀 Processing Game: {game}")
        train_and_evaluate(game)
        
    print("🎉 Machine Learning Pipeline Complete!")