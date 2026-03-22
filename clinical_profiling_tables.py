import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def coerce_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['status', 'Group_Status', 'ID']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(' ', ''), errors='coerce')
    return df

def categorize_features(columns):
    """Groups tablet variables into readable clinical domains."""
    domains = {
        'Speed & Reaction': [c for c in columns if any(x in c.lower() for x in ['time', 'duration'])],
        'Touch Pressure': [c for c in columns if 'pressure' in c.lower()],
        'Accuracy & Errors': [c for c in columns if any(x in c.lower() for x in ['correct', 'outside', 'not_completed', 'no_taps'])],
        'Complex Kinematics': [c for c in columns if any(x in c.lower() for x in ['drag', 'pinch', 'multi_finger', 'number_taps'])]
    }
    return domains

def generate_clinical_profiles(csv_file, dataset_name, healthy_label='healthy', threshold=80):
    print(f"\n{'='*60}")
    print(f"--- Generating Extreme Clinical Phenotypes Table (> {threshold}%): {dataset_name} ---")
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return

    df = coerce_numeric(df)
    target_col = 'status' if 'status' in df.columns else 'Group_Status'
    
    test_cols = [c for c in df.columns if c.startswith(('T1_', 'T2_', 'T3_', 'T5_', 'T6_', 'T7_1_', 'T7_2_', 'T10_', 'T12_', 'T13_', 'T20_'))]
    domains = categorize_features(test_cols)
    
    actual_healthy_label = next((s for s in df[target_col].dropna().unique() if healthy_label.lower() in s.lower()), None)
    
    if not actual_healthy_label:
        print(f"Could not find a 'healthy' group in {target_col}.")
        return

    statuses = df[target_col].dropna().unique()
    
    profiles = {}
    for status in statuses:
        status_df = df[df[target_col] == status]
        profiles[status] = status_df[test_cols].median()

    healthy_baseline = profiles[actual_healthy_label]
    table_data = []

    for status in statuses:
        if status == actual_healthy_label:
            continue
            
        status_medians = profiles[status]
        
        for domain_name, features in domains.items():
            for feat in features:
                h_val = healthy_baseline[feat]
                s_val = status_medians[feat]
                
                if pd.notna(h_val) and pd.notna(s_val) and h_val != 0:
                    pct_change = ((s_val - h_val) / h_val) * 100
                    
                    if abs(pct_change) >= threshold:
                        clean_feat = feat.replace('_DOM_', '\n').replace('_', ' ')
                        direction = "Increased" if pct_change > 0 else "Decreased"
                        
                        table_data.append([
                            status,
                            domain_name,
                            clean_feat,
                            direction,
                            f"{pct_change:+.1f}%",
                            f"{s_val:.2f}",
                            f"{h_val:.2f}"
                        ])
                        
    columns = ['Clinical Status', 'Feature Category', 'Tablet Feature', 'Direction', '% Change', 'Patient Median', 'Healthy Baseline']
    results_df = pd.DataFrame(table_data, columns=columns)
    
    if results_df.empty:
        print(f"No features found with >{threshold}% deviation for {dataset_name}.")
        return
        
    # 1. Save CSV
    csv_filename = f"{dataset_name}_Extreme_Deviations_Table.csv"
    results_df.to_csv(csv_filename, index=False)
    
    # 2. GENERATE MATPLOTLIB FIGURE TABLE
    # Dynamically scale the image height based on the number of rows
    fig_height = len(results_df) * 0.6 + 1.5 
    fig, ax = plt.subplots(figsize=(14, fig_height))
    
    # Hide axes
    ax.axis('off')
    ax.axis('tight')
    
    # Draw the table
    table = ax.table(cellText=results_df.values, 
                     colLabels=results_df.columns, 
                     loc='center', 
                     cellLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0) # Stretch the cells so text fits nicely
    
    # Color-code the headers and alternate row colors
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4C72B0') # Nice professional blue
        else:
            if row % 2 == 0:
                cell.set_facecolor('#F2F2F2') # Light grey alternating rows
                
            # Make the % Change column bold
            if col == 4:
                cell.set_text_props(weight='bold')
                # Color code Increase (Red) vs Decrease (Green/Blue) if you like, 
                # but leaving it bold is usually cleanest!
    
    plt.title(f"Extreme Clinical Deviations (> {threshold}%): {dataset_name}\nBaseline: '{actual_healthy_label}'", 
              fontweight="bold", size=14, y=1.0)
    
    png_filename = f"{dataset_name}_Extreme_Deviations_Figure.png"
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    
    print(f"Saved Figure Table to: '{png_filename}'\n")

if __name__ == "__main__":
    run_elderly = True
    run_children = True
    
    if run_elderly:
        generate_clinical_profiles('data/Eldery_Final_ML_Ready.csv', 'Elderly', healthy_label='healthy', threshold=80)
    if run_children:
        generate_clinical_profiles('data/Children_Final_ML_Ready.csv', 'Children', healthy_label='healthy', threshold=80)