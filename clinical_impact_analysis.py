import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

def coerce_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['status', 'Group_Status', 'ID']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(' ', ''), errors='coerce')
    return df

def run_imputed_impact_analysis(csv_file, dataset_name, demographics_list, tests_to_include):
    print(f"\n{'='*60}")
    print(f"--- Running Imputed Clinical Impact Analysis: {dataset_name} ---")
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return

    df = coerce_numeric(df)
    target_col = 'status' if 'status' in df.columns else 'Group_Status'
    
    # 1. FIX THE DATA LEAK: Neutralize the -1 in Frequency_Tablet_Use
    if 'Frequency_Tablet_Use' in df.columns:
        print("Neutralizing artificial '-1' values in Frequency_Tablet_Use...")
        # Replace -1 with NaN temporarily
        df['Frequency_Tablet_Use'] = df['Frequency_Tablet_Use'].replace(-1, np.nan)
        # Fill those NaNs with the overall median so the AI can't use it as a cheat code
        df['Frequency_Tablet_Use'] = df['Frequency_Tablet_Use'].fillna(df['Frequency_Tablet_Use'].median())

    # 2. Filter for specifically requested tests
    print(f"Filtering strictly for tests: {tests_to_include}")
    test_prefixes = tuple([f"{t}_" for t in tests_to_include])
    test_cols = [c for c in df.columns if c.startswith(test_prefixes)]
    
    if not test_cols:
        print("Error: No columns found for these tests!")
        return

    valid_demographics = [d for d in demographics_list if d in df.columns]
    
    # 3. IMPUTATION: Fill missing tablet data with the median (for Parkinson's skipped tests)
    print("Imputing missing tablet test data with the median...")
    for col in test_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Safely drop any rows where the target status or OTHER demographics are missing
    df = df.dropna(subset=[target_col] + valid_demographics)
    
    X = df[test_cols + valid_demographics].select_dtypes(include=['float64', 'int64']).copy()
    y = df[target_col].copy()
    
    # 4. Train XGBoost
    print("Training XGBoost Classifier...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb.fit(X, y_encoded)
    
    importances = xgb.feature_importances_
    feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    
    # ==========================================
    # PART A: DEMOGRAPHIC CONFOUNDER CHECK
    # ==========================================
    top_15_overall = feat_df.sort_values(by='Importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    colors = ['#e74c3c' if feat in valid_demographics else '#3498db' for feat in top_15_overall['Feature']]
    
    sns.barplot(data=top_15_overall, x='Importance', y='Feature', palette=colors)
    plt.title(f'Demographic Confounder Check (Imputed Data)\n({dataset_name} - Red = Demographic, Blue = Tablet)', fontsize=14)
    plt.xlabel('XGBoost Importance Score')
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_Imputed_Demographic_Check.png', dpi=300)
    print(f"Saved: {dataset_name}_Imputed_Demographic_Check.png")

    # ==========================================
    # PART B: ENTIRE TEST AGGREGATION (T1 vs T5, etc.)
    # ==========================================
    test_only_df = feat_df[~feat_df['Feature'].isin(valid_demographics)].copy()
    test_only_df['Test_Name'] = test_only_df['Feature'].apply(lambda x: x.split('_')[0])
    
    test_impact = test_only_df.groupby('Test_Name')['Importance'].sum().reset_index()
    test_impact = test_impact.sort_values(by='Importance', ascending=False)
    
    total_test_importance = test_impact['Importance'].sum()
    test_impact['Importance_Pct'] = (test_impact['Importance'] / total_test_importance) * 100
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=test_impact, x='Importance_Pct', y='Test_Name', palette='magma')
    plt.title(f'Overall Test Importance: Which game is most valuable?\n({dataset_name} Imputed Data)', fontsize=14)
    plt.xlabel('Total Contribution to Diagnosis (%)')
    plt.ylabel('Tablet Test')
    
    for index, value in enumerate(test_impact['Importance_Pct']):
        plt.text(value + 0.5, index, f'{value:.1f}%', va='center')
        
    plt.xlim(0, test_impact['Importance_Pct'].max() + 15)
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_Imputed_Test_Importance.png', dpi=300)
    print(f"Saved: {dataset_name}_Imputed_Test_Importance.png\n")


if __name__ == "__main__":
    
    # Exactly the tests you requested
    target_tests = ['T1', 'T2', 'T3', 'T12', 'T13', 'T20']
    
    # 1. Elderly Data 
    elderly_demographics = ['Age', 'Gender', 'Education', 'Frequency_Tablet_Use']
    run_imputed_impact_analysis('data/Eldery_Final_ML_Ready.csv', 'Elderly', elderly_demographics, target_tests)
    
    # 2. Children Data
    children_demographics = ['Age', 'Gender', 'Frequency_Tablet_Use']
    run_imputed_impact_analysis('data/Children_Final_ML_Ready.csv', 'Children', children_demographics, target_tests)