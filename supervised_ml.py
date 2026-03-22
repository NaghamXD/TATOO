import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

def coerce_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['status', 'Group_Status', 'ID']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(' ', ''), errors='coerce')
    return df

def tune_models_for_recall(X, y_encoded):
    print("\n[ Phase 1: Hunting for the Best Recall Parameters ]")
    
    # Base Pipelines
    rf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    xgb_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
    ])

    # Parameter Grids (Kept concise to save computing time)
    rf_param_grid = {
        'classifier__n_estimators': [100, 300],
        'classifier__max_depth': [None, 5, 10],
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    }

    xgb_param_grid = {
        'classifier__n_estimators': [100, 300],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.01, 0.05, 0.1]
    }

    # GridSearchCV hunting specifically for 'recall_macro'
    rf_search = GridSearchCV(rf_pipeline, rf_param_grid, cv=3, scoring='recall_macro', n_jobs=-1)
    xgb_search = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=3, scoring='recall_macro', n_jobs=-1)

    print("  -> Tuning Random Forest (This might take a minute)...")
    rf_search.fit(X, y_encoded)
    print(f"  -> Best RF Macro Recall Found: {rf_search.best_score_ * 100:.2f}%")

    print("  -> Tuning XGBoost (This might take a minute)...")
    xgb_search.fit(X, y_encoded)
    print(f"  -> Best XGB Macro Recall Found: {xgb_search.best_score_ * 100:.2f}%")
    
    # Return the perfectly tuned models to use in Phase 2
    return rf_search.best_estimator_, xgb_search.best_estimator_

def run_classifiers(csv_file, dataset_name):
    print(f"\n{'='*60}")
    print(f"--- Processing Dataset: {dataset_name} ---")
    print(f"{'='*60}")
    
    # 1. Load Data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return

    df = coerce_numeric(df)
    target_col = 'status' if 'status' in df.columns else 'Group_Status'
    
    test_cols = [c for c in df.columns if c.startswith(('T1_', 'T2_', 'T5_', 'T12_', 'T13_', 'T20_'))]
    X = df[test_cols].select_dtypes(include=['float64', 'int64']).copy()
    y = df[target_col].copy()
    
    # 2. Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    
    # 3. PHASE 1: Tune the Models for Maximum Recall
    best_rf_pipeline, best_xgb_pipeline = tune_models_for_recall(X, y_encoded)
    
    # 4. PHASE 2: 5-Fold Cross Validation
    print("\n[ Phase 2: Running 5-Fold Cross-Validation with Tuned Models ]")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    y_true_all, rf_preds_all, xgb_preds_all = [], [], []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y_encoded)):
        print(f"  -> Processing Fold {fold + 1}/5...")
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]
        
        # Train & Predict using the TUNED models
        best_rf_pipeline.fit(X_train, y_train)
        rf_preds = best_rf_pipeline.predict(X_test)
        
        best_xgb_pipeline.fit(X_train, y_train)
        xgb_preds = best_xgb_pipeline.predict(X_test)
        
        y_true_all.extend(y_test)
        rf_preds_all.extend(rf_preds)
        xgb_preds_all.extend(xgb_preds)

    # 5. Evaluate Aggregated Results
    print(f"\n{'='*40}")
    print(f"FINAL AGGREGATED RESULTS: {dataset_name}")
    print(f"{'='*40}")
    
    print("\n[ Optimized Random Forest Classification Report ]")
    print(classification_report(y_true_all, rf_preds_all, target_names=class_names))

    print("\n[ Optimized XGBoost Classification Report ]")
    print(classification_report(y_true_all, xgb_preds_all, target_names=class_names))

    # 6. Plot Confusion Matrices
    cm_xgb = confusion_matrix(y_true_all, xgb_preds_all)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Tuned XGBoost Overall Confusion Matrix ({dataset_name})', fontsize=16)
    plt.ylabel('Actual Clinical Status', fontsize=12)
    plt.xlabel('Tablet Predicted Status', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_Tuned_CV_Confusion_Matrix_XGBoost.png", dpi=300)

    cm_rf = confusion_matrix(y_true_all, rf_preds_all)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Tuned RandomForest Overall Confusion Matrix ({dataset_name})', fontsize=16)
    plt.ylabel('Actual Clinical Status', fontsize=12)
    plt.xlabel('Tablet Predicted Status', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_Tuned_CV_Confusion_Matrix_RandomForest.png", dpi=300)

if __name__ == "__main__":
    run_classifiers('data/Eldery_Final_ML_Ready.csv', 'Elderly')
    run_classifiers('data/Children_Final_ML_Ready.csv', 'Children')