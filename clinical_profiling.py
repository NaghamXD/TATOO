import pandas as pd
import numpy as np
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

def generate_clinical_profiles(csv_file, dataset_name, healthy_label='healthy'):
    print(f"\n{'='*60}")
    print(f"--- Generating Clinical Phenotypes: {dataset_name} ---")
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return

    df = coerce_numeric(df)
    target_col = 'status' if 'status' in df.columns else 'Group_Status'
    
    # Identify all tablet features
    test_cols = [c for c in df.columns if c.startswith(('T1_', 'T2_', 'T3_', 'T5_', 'T6_', 'T7_1_', 'T7_2_', 'T10_', 'T12_', 'T13_', 'T20_'))]
    domains = categorize_features(test_cols)
    
    # Find the exact healthy label in the dataset
    actual_healthy_label = next((s for s in df[target_col].dropna().unique() if healthy_label.lower() in s.lower()), None)
    
    if not actual_healthy_label:
        print(f"Could not find a 'healthy' group in {target_col} to use as a baseline.")
        return

    statuses = df[target_col].dropna().unique()
    
    # Calculate Medians for every group (Medians are safer than Means against extreme outliers)
    profiles = {}
    for status in statuses:
        status_df = df[df[target_col] == status]
        profiles[status] = status_df[test_cols].median()

    healthy_baseline = profiles[actual_healthy_label]
    
    report_lines = []
    report_lines.append(f"CLINICAL PHENOTYPE REPORT: {dataset_name.upper()}")
    report_lines.append(f"Baseline for comparison: '{actual_healthy_label}'")
    report_lines.append("="*60 + "\n")

    for status in statuses:
        if status == actual_healthy_label:
            continue # Skip comparing healthy to healthy
            
        report_lines.append(f"► PROFILE: {status.upper()}")
        report_lines.append(f"  (Based on tests performed by this cohort)")
        
        status_medians = profiles[status]
        
        for domain_name, features in domains.items():
            report_lines.append(f"\n  [{domain_name}]")
            
            # Find the top 3 features in this domain with the biggest % difference from Healthy
            diffs = []
            for feat in features:
                h_val = healthy_baseline[feat]
                s_val = status_medians[feat]
                
                # Only compare if both the disease group and healthy group actually played this test
                if pd.notna(h_val) and pd.notna(s_val) and h_val != 0:
                    pct_change = ((s_val - h_val) / h_val) * 100
                    
                    # Ignore tiny differences (less than 5% change) to keep the report clean
                    if abs(pct_change) > 5:
                        diffs.append((feat, pct_change, s_val, h_val))
            
            # Sort by absolute magnitude of difference
            diffs.sort(key=lambda x: abs(x[1]), reverse=True)
            
            if not diffs:
                report_lines.append("    - No significant deviations from healthy baseline.")
            else:
                for feat, pct_change, s_val, h_val in diffs[:4]: # Show top 4 defining traits per domain
                    direction = "Increased" if pct_change > 0 else "Decreased"
                    
                    # Make the text readable
                    clean_feat = feat.replace('_DOM_', ' ').replace('_', ' ')
                    report_lines.append(f"    - {clean_feat}: {direction} by {abs(pct_change):.1f}% (Patient: {s_val:.2f} vs Healthy: {h_val:.2f})")
        
        report_lines.append("\n" + "-"*60 + "\n")

    # Save to text file
    output_filename = f"{dataset_name}_Clinical_Profiles.txt"
    with open(output_filename, 'w') as f:
        f.write("\n".join(report_lines))
        
    print(f"Success! Generated readable clinical profiles: '{output_filename}'")
    
    # Also save the raw medians to a CSV if you want to make your own graphs later
    median_df = pd.DataFrame(profiles).T
    csv_filename = f"{dataset_name}_Profile_Medians.csv"
    median_df.to_csv(csv_filename)
    print(f"Saved raw median data to: '{csv_filename}'")

if __name__ == "__main__":
    run_elderly = True
    run_children = True
    
    if run_elderly:
        generate_clinical_profiles('data/Eldery_Final_ML_Ready.csv', 'Elderly', healthy_label='healthy')
    if run_children:
        generate_clinical_profiles('data/Children_Final_ML_Ready.csv', 'Children', healthy_label='healthy')