import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

def coerce_numeric(df):
    """Ensures all game columns are strictly numeric, fixing any dirty string spacing."""
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['status', 'Group_Status', 'ID']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(' ', ''), errors='coerce')
    return df

def get_target_col(df):
    """Finds the correct target column depending on the dataset version."""
    if 'status' in df.columns: return 'status'
    elif 'Group_Status' in df.columns: return 'Group_Status'
    return None

def create_eda_summaries(input_file, output_excel):
    print(f"\n--- Generating EDA Summary for {input_file} ---")
    df = pd.read_csv(input_file)
    df = coerce_numeric(df)
    target_col = get_target_col(df)
    
    if not target_col:
        print(f"Error: No target column ('status') found in {input_file}")
        return
        
    writer = pd.ExcelWriter(output_excel, engine='xlsxwriter')
    
    # ==========================================
    # 1. Demographics Summary Table
    # ==========================================
    demo_cols = ['Age', 'Gender', 'Education', 'Frequency_Tablet_Use']
    demo_cols = [c for c in demo_cols if c in df.columns]
    
    demo_summary = []
    for stat in sorted(df[target_col].dropna().unique()):
        stat_df = df[df[target_col] == stat]
        n_subjects = len(stat_df)
        
        row_dict = {'Status': stat, 'Total_Subjects': n_subjects}
        if 'Age' in demo_cols:
            row_dict['Age (Mean)'] = round(stat_df['Age'].mean(), 1)
            row_dict['Age (Range)'] = f"{stat_df['Age'].min()} to {stat_df['Age'].max()}"
            
        if 'Gender' in demo_cols:
            # Captures breakdown (e.g., how many 1s vs 2s)
            g_counts = stat_df['Gender'].value_counts().to_dict()
            row_dict['Gender (Breakdown)'] = str(g_counts)
            
        if 'Education' in demo_cols:
            row_dict['Education (Mean Yrs)'] = round(stat_df['Education'].mean(), 1)
            row_dict['Education (Range)'] = f"{stat_df['Education'].min()} to {stat_df['Education'].max()}"
            
        if 'Frequency_Tablet_Use' in demo_cols:
            f_counts = stat_df['Frequency_Tablet_Use'].value_counts().to_dict()
            row_dict['Freq_Tablet_Use (Breakdown)'] = str(f_counts)
            
        demo_summary.append(row_dict)
        
    pd.DataFrame(demo_summary).to_excel(writer, sheet_name='Demographics', index=False)
    
    # ==========================================
    # 2. Clinical Tests Summary Tables
    # ==========================================
    test_cols = [c for c in df.columns if re.match(r'^T\d+_', c)]
    test_prefixes = sorted(list(set([re.match(r'^(T\d+)_', c).group(1) for c in test_cols])), key=lambda x: int(x[1:]))
    
    # Keywords mapping to your requested OT variables
    metric_keywords = [
        'reaction_time', 'flight_time', 'touch_time', 'duration',
        'low_pressure', 'medium_pressure', 'high_pressure',
        'number_taps', 'attempts', 'touch_outside', 
        'drag_completed', 'drag_not_completed', 'pinch_completed', 'pinch_not_completed'
    ]
    
    for test in test_prefixes:
        c_for_test = [c for c in test_cols if c.startswith(f"{test}_")]
        
        # Filter for relevant metrics (case insensitive)
        relevant_cols = [c for c in c_for_test if any(kw in c.lower() for kw in metric_keywords)]
        if not relevant_cols: continue
        
        test_summary = []
        for stat in sorted(df[target_col].dropna().unique()):
            stat_df = df[df[target_col] == stat]
            
            for col in relevant_cols:
                series = stat_df[col].dropna()
                if len(series) > 0:
                    test_summary.append({
                        'Status': stat,
                        'OT_Variable': col,
                        'Mean': round(series.mean(), 2),
                        'Median': round(series.median(), 2),
                        'Range': f"{round(series.min(), 2)} to {round(series.max(), 2)}"
                    })
                else:
                    test_summary.append({
                        'Status': stat,
                        'OT_Variable': col,
                        'Mean': 'N/A', 'Median': 'N/A', 'Range': 'N/A'
                    })
                
        test_df = pd.DataFrame(test_summary)
        
        # Pivot the table so variables are rows and stats are grouped by Status columns
        if not test_df.empty:
            pivot_df = test_df.pivot(index='OT_Variable', columns='Status', values=['Mean', 'Median', 'Range'])
            pivot_df.to_excel(writer, sheet_name=test)
            
    writer.close()
    print(f"Exported successfully to {output_excel}")

# Run for both finalized datasets
if __name__ == "__main__":
    create_eda_summaries('Eldery_Final_ML_Ready.csv', 'Elderly_EDA_Summary.xlsx')
    create_eda_summaries('Children_Final_ML_Ready.csv', 'Children_EDA_Summary.xlsx')