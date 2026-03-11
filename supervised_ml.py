import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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

def run_classifiers(csv_file, dataset_name):
    print(f"\n{'='*60}")
    print(f"--- Training Supervised ML Models: {dataset_name} ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return

    df = coerce_numeric(df)
    target_col = 'status' if 'status' in df.columns else 'Group_Status'
    
    # Use all numeric test features
    test_cols = [c for c in df.columns if c.startswith(('T1_', 'T2_', 'T5_', 'T12_', 'T13_', 'T20_'))]
    X = df[test_cols].select_dtypes(include=['float64', 'int64']).copy()
    y = df[target_col].copy()
    
    # 2. Encode Labels (Healthy -> 0, Parkinson -> 1, etc.)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    
    # 3. Train/Test Split (Stratified ensures equal class distribution)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # 4. Create the ML Pipelines
    # Pipeline safely imputes missing data using the median, then scales the data!
    rf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42))
    ])

    xgb_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric='mlogloss', random_state=42))
    ])

    # 5. Train & Evaluate Random Forest
    print("\n[ Training Random Forest... ]")
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    print(classification_report(y_test, rf_preds, target_names=class_names))

    # 6. Train & Evaluate XGBoost
    print("\n[ Training XGBoost... ]")
    xgb_pipeline.fit(X_train, y_train)
    xgb_preds = xgb_pipeline.predict(X_test)
    print(classification_report(y_test, xgb_preds, target_names=class_names))

    # 7. Plot XGBoost Confusion Matrix
    # A confusion matrix shows exactly WHICH diseases the tablet got confused by
    cm = confusion_matrix(y_test, xgb_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'XGBoost Confusion Matrix ({dataset_name})\n(Where did the model make mistakes?)', fontsize=16)
    plt.ylabel('Actual Clinical Status', fontsize=12)
    plt.xlabel('Tablet Predicted Status', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    cm_filename = f"{dataset_name}_Confusion_Matrix_XGBoost.png"
    plt.savefig(cm_filename, dpi=300)
    print(f"Saved Confusion Matrix to '{cm_filename}'")

    # 8. Plot Random Forest Confusion Matrix
    # A confusion matrix shows exactly WHICH diseases the tablet got confused by
    cm1 = confusion_matrix(y_test, rf_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'RandomForest Confusion Matrix ({dataset_name})\n(Where did the model make mistakes?)', fontsize=16)
    plt.ylabel('Actual Clinical Status', fontsize=12)
    plt.xlabel('Tablet Predicted Status', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    cm1_filename = f"{dataset_name}_Confusion_Matrix_RandomForest.png"
    plt.savefig(cm1_filename, dpi=300)
    print(f"Saved Confusion Matrix to '{cm1_filename}'")

if __name__ == "__main__":
    run_classifiers('data/Eldery_Final_ML_Ready.csv', 'Elderly')
    run_classifiers('data/Children_Final_ML_Ready.csv', 'Children')