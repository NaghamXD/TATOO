import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

def coerce_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['status', 'Group_Status', 'ID']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(' ', ''), errors='coerce')
    return df

def run_ab_test(csv_file, dataset_name, demographics_list, target_tests):
    print(f"\n{'='*60}")
    print(f"--- Running A/B Demographic Test: {dataset_name} ---")
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return

    df = coerce_numeric(df)
    target_col = 'status' if 'status' in df.columns else 'Group_Status'
    
    # 1. FIX THE DATA LEAK: Neutralize the -1 in Frequency_Tablet_Use
    if 'Frequency_Tablet_Use' in df.columns:
        df['Frequency_Tablet_Use'] = df['Frequency_Tablet_Use'].replace(-1, np.nan)
        df['Frequency_Tablet_Use'] = df['Frequency_Tablet_Use'].fillna(df['Frequency_Tablet_Use'].median())

    # 2. Filter for specifically requested tests
    test_prefixes = tuple([f"{t}_" for t in target_tests])
    test_cols = [c for c in df.columns if c.startswith(test_prefixes)]
    valid_demographics = [d for d in demographics_list if d in df.columns]

    # 3. IMPUTATION: Fill missing tablet data with the median
    for col in test_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # Also safely impute any remaining NaN in demographics just in case
    for col in valid_demographics:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Drop rows without a target label
    df = df.dropna(subset=[target_col])
    
    # Define our two feature sets
    X_tablet_only = df[test_cols].select_dtypes(include=['float64', 'int64']).copy()
    X_with_demos = df[test_cols + valid_demographics].select_dtypes(include=['float64', 'int64']).copy()
    y = df[target_col].copy()
    
    # Encode Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Stratified Split (Keep the exact same split for both tests for fairness)
    indices = np.arange(len(y_encoded))
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(indices, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Scale both sets based on their respective training sets
    scaler_A = StandardScaler()
    X_train_A = scaler_A.fit_transform(X_tablet_only.iloc[X_train_idx])
    X_test_A = scaler_A.transform(X_tablet_only.iloc[X_test_idx])
    
    scaler_B = StandardScaler()
    X_train_B = scaler_B.fit_transform(X_with_demos.iloc[X_train_idx])
    X_test_B = scaler_B.transform(X_with_demos.iloc[X_test_idx])

    # Initialize Models
    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric='mlogloss', random_state=42)

    results = []

    # --- Train SET A (Tablet Only) ---
    rf.fit(X_train_A, y_train)
    xgb.fit(X_train_A, y_train)
    
    results.append({'Model': 'Random Forest', 'Features': 'Tablet Tests ONLY', 
                    'Accuracy': accuracy_score(y_test, rf.predict(X_test_A)),
                    'Macro_F1': f1_score(y_test, rf.predict(X_test_A), average='macro')})
                    
    results.append({'Model': 'XGBoost', 'Features': 'Tablet Tests ONLY', 
                    'Accuracy': accuracy_score(y_test, xgb.predict(X_test_A)),
                    'Macro_F1': f1_score(y_test, xgb.predict(X_test_A), average='macro')})

    # --- Train SET B (Tablet + Demographics) ---
    rf.fit(X_train_B, y_train)
    xgb.fit(X_train_B, y_train)
    
    results.append({'Model': 'Random Forest', 'Features': 'Tablet + Demographics', 
                    'Accuracy': accuracy_score(y_test, rf.predict(X_test_B)),
                    'Macro_F1': f1_score(y_test, rf.predict(X_test_B), average='macro')})
                    
    results.append({'Model': 'XGBoost', 'Features': 'Tablet + Demographics', 
                    'Accuracy': accuracy_score(y_test, xgb.predict(X_test_B)),
                    'Macro_F1': f1_score(y_test, xgb.predict(X_test_B), average='macro')})

    results_df = pd.DataFrame(results)
    
    # Print summary to terminal
    print("\n[ Performance Summary ]")
    print(results_df.to_string(index=False))

    # --- Plotting the A/B Test ---
    plt.figure(figsize=(10, 6))
    
    # We will plot Accuracy. (You can change 'Accuracy' to 'Macro_F1' if you prefer)
    sns.barplot(data=results_df, x='Model', y='Accuracy', hue='Features', palette=['#3498db', '#e74c3c'])
    
    plt.title(f'A/B Model Ablation Test: Impact of Demographics\n({dataset_name} Data | Tests: {", ".join(target_tests)})', fontsize=14)
    plt.ylabel('Diagnostic Accuracy')
    plt.ylim(0, 1.0)
    
    # Add percentage labels on top of the bars
    for p in plt.gca().patches:
        plt.gca().annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', fontsize=11, color='black', xytext=(0, 8), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(f'{dataset_name}_AB_Demographic_Test.png', dpi=300)
    print(f"\nSaved comparison chart to '{dataset_name}_AB_Demographic_Test.png'")

if __name__ == "__main__":
    
    target_tests = ['T1', 'T2', 'T5', 'T12', 'T13', 'T20']
    
    # 1. Elderly Data
    elderly_demographics = ['Age', 'Gender', 'Education', 'Frequency_Tablet_Use']
    run_ab_test('data/Eldery_Final_ML_Ready.csv', 'Elderly', elderly_demographics, target_tests)
    
    # 2. Children Data
    children_demographics = ['Age', 'Gender', 'Frequency_Tablet_Use']
    run_ab_test('data/Children_Final_ML_Ready.csv', 'Children', children_demographics, target_tests)