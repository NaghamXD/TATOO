import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
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
MODEL_PERFORMANCE_DIR = os.path.join(RESULTS_DIR, "model_performance")
TARGET_COLUMN = 'Target_Age_Old'

for _dir in (FEATURE_IMPORTANCE_DIR, MODEL_PERFORMANCE_DIR):
    if not os.path.exists(_dir):
        os.makedirs(_dir)

def train_and_evaluate(game_name):
    file_path = os.path.join(PROCESSED_DIR, f"MASTER_{game_name}.csv")
    if not os.path.exists(file_path):
        return None
        
    df = pd.read_csv(file_path)
    
    # 1. Clean Data
    df = df.dropna(subset=[TARGET_COLUMN])
    if df.empty or len(df) < 10:
        print(f"⚠️ Not enough labeled data to train {game_name}.")
        return None

    # 2. Separate Features (X) from Target (y)
    cols_to_drop = [
        'Patient_ID', 'Age', 'Target_Age_Old', 'Target_Visual_Impaired',
        'Correct_Taps', 'Outside_Taps', 'Flight_Duration_sec', 
        'Touch_Duration_sec', 'Gaze_Pct_Within_Screen', 'Gaze_Pct_Outside_Screen',
        'Pressure_Med_Pct', 'Game'
    ]
    
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    y = df[TARGET_COLUMN]
    patient_ids = df['Patient_ID'] # Keep IDs to map results back to users

    X = X.apply(pd.to_numeric, errors='coerce')

    # 3. Initialize Model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42
    )

    # 4. Cross-Validation Scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"✅ {game_name} Accuracy: {accuracies.mean()*100:.1f}%")

    # 5. Generate Individual Patient Predictions (The "Confusion" Data)
    # This gives us a prediction for every patient using the CV folds
    y_pred = cross_val_predict(model, X, y, cv=cv)
    
    # Compare prediction to actual label (1 = Correct, 0 = Incorrect)
    is_correct = (y_pred == y).astype(int)
    
    # Store results in a series indexed by Patient_ID
    patient_performance = pd.Series(is_correct.values, index=patient_ids, name=game_name)

    # 6. Generate Feature Importance Plot (Existing logic)
    model.fit(X, y)
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title(f"Top Biomarkers: {game_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FEATURE_IMPORTANCE_DIR, f"{game_name}_Importance.png"), dpi=300)
    plt.close()
    
    return patient_performance

if __name__ == "__main__":
    TARGET_GAMES = ['TouchIt', 'CornerIt', 'DoubleTapIt', 'PinchIt', 'SlideIt', 'DragIt']
    
    print(f"\n{'='*60}")
    print(f"🧠 TRAINING CLINICAL MODEL & GENERATING PERFORMANCE MATRIX")
    print(f"{'='*60}\n")
    
    all_game_results = []
    
    for game in TARGET_GAMES:
        res = train_and_evaluate(game)
        if res is not None:
            all_game_results.append(res)
            
    # --- GENERATE THE PATIENT-GAME MATRIX ---
    # Combine all series into one DataFrame where rows=Patients, columns=Games
    perf_matrix = pd.concat(all_game_results, axis=1)
    
    # Sort by Patient ID for a clean view
    perf_matrix = perf_matrix.sort_index()

    # Create the Heatmap Visualization
    plt.figure(figsize=(12, 14))
    sns.heatmap(perf_matrix, annot=True, cmap="RdYlGn", cbar=False, linewidths=.5)
    plt.title(f"User Performance Matrix (1=Correct Prediction, 0=Incorrect)\nTarget: {TARGET_COLUMN}")
    plt.ylabel("Patient ID")
    plt.xlabel("Game Type")
    plt.tight_layout()
    
    matrix_plot_path = os.path.join(MODEL_PERFORMANCE_DIR, "User_Performance_Matrix.png")
    plt.savefig(matrix_plot_path, dpi=300)

    # Save as CSV for detailed analysis
    perf_matrix.to_csv(os.path.join(MODEL_PERFORMANCE_DIR, "User_Performance_Matrix.csv"))
    
    print(f"\n🎉 Pipeline Complete!")
    print(f"🖼️  Saved Performance Matrix to: {matrix_plot_path}")
    print(f"📊 Saved Raw Matrix Data to: {RESULTS_DIR}User_Performance_Matrix.csv")