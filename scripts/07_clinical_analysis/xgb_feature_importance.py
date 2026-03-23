import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
import re

warnings.filterwarnings('ignore')

def coerce_numeric(df):
    """Cleans up any dirty strings to pure numbers."""
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['status', 'Group_Status', 'ID']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(' ', ''), errors='coerce')
    return df

def run_xgboost_importance(csv_file, dataset_name, tests_to_include=None, exclude_groups=None):
    print(f"\n{'='*60}")
    print(f"--- Extracting XGBoost Feature Importance: {dataset_name} ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return

    df = coerce_numeric(df)
    target_col = 'status' if 'status' in df.columns else 'Group_Status'
    
    # 2. Exclude specified groups (e.g., if you want to drop Parkinson's for a Deep Dive)
    if exclude_groups:
        initial_len = len(df)
        df = df[~df[target_col].isin(exclude_groups)]
        print(f"Excluded groups: {exclude_groups}. Dropped {initial_len - len(df)} subjects.")

    # 3. Isolate features from the SPECIFIC tests requested
    if tests_to_include:
        print(f"Restricting analysis strictly to tests: {tests_to_include}")
        test_prefixes = tuple([f"{t}_" for t in tests_to_include])
        test_cols = [c for c in df.columns if c.startswith(test_prefixes)]
    else:
        print("Analyzing ALL available tests.")
        test_cols = [c for c in df.columns if re.match(r'^T\d+_', c)]
    
    if not test_cols:
        print("Error: No columns found for the requested tests!")
        return

    # Keep only numeric columns for XGBoost
    X = df[test_cols].select_dtypes(include=['float64', 'int64']).copy()
    y = df[target_col].copy()
    
    print(f"Total subjects: {len(X)}")
    print(f"Total features analyzed: {len(X.columns)}")

    # 4. Encode the target clinical labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 5. Train XGBoost
    print("Training XGBoost Classifier...")
    xgb = XGBClassifier(
        n_estimators=300, 
        max_depth=5, 
        learning_rate=0.05, 
        use_label_encoder=False, 
        eval_metric='mlogloss', 
        random_state=42
    )
    xgb.fit(X, y_encoded)
    
    # 6. Extract and Sort Feature Importances
    importances = xgb.feature_importances_
    
    feat_imp_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    })
    
    # Drop features that had exactly 0 importance to clean up the plot
    feat_imp_df = feat_imp_df[feat_imp_df['Importance'] > 0]
    top_15_features = feat_imp_df.sort_values(by='Importance', ascending=False).head(15)
    
    # 7. Plotting
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=top_15_features, 
        x='Importance', 
        y='Feature', 
        palette='viridis'
    )
    
    # Create a dynamic title based on the tests included
    scenario_str = "All Tests" if not tests_to_include else ", ".join(tests_to_include)
    plt.title(f'Top 15 Most Important Features ({dataset_name} | {scenario_str})\nCalculated by XGBoost Information Gain', fontsize=16)
    plt.xlabel('Importance Score (Higher = Better at diagnosing the disease)', fontsize=12)
    plt.ylabel('Tablet Variables', fontsize=12)
    
    plt.tight_layout()
    
    # Clean up filename
    safe_scenario_name = scenario_str.replace(', ', '_')
    filename = f"{dataset_name}_{safe_scenario_name}_XGBoost_Importance.png"
    plt.savefig(filename, dpi=300)
    print(f"Success! Saved feature importance chart to '{filename}'")
    
    # Print the absolute Top 4
    print(f"\n--- 🏆 THE TOP 4 FEATURES FOR {dataset_name.upper()} ({scenario_str}) ---")
    top_4_list = []
    for i, feat in enumerate(top_15_features['Feature'].head(12)):
        print(f"  {i+1}. {feat}")
        top_4_list.append(feat)
        
    print("\n(Copy these variable names into your clustering script!)")

if __name__ == "__main__":
    # 1. Elderly: The Universal Battery (Tests everyone played)
    run_xgboost_importance(
        csv_file='data/Eldery_Final_ML_Ready.csv', 
        dataset_name='Elderly', 
        tests_to_include=['T5', 'T13', 'T20'],
        exclude_groups=None
    )
    
    # 2. Elderly: The Deep Dive (All core tests, excluding Parkinson's)
    run_xgboost_importance(
        csv_file='data/Eldery_Final_ML_Ready.csv', 
        dataset_name='Elderly', 
        tests_to_include=['T1', 'T2', 'T5', 'T12', 'T13', 'T20'],
        exclude_groups=['Parkinson']
    )

    # 3. Children: Core Battery
    run_xgboost_importance(
        csv_file='data/Children_Final_ML_Ready.csv', 
        dataset_name='Children', 
        tests_to_include=['T1', 'T2', 'T5', 'T12', 'T13', 'T20'],
        exclude_groups=None
    )