import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re

# Ignore cosmetic warnings from Seaborn
warnings.filterwarnings('ignore')

def coerce_numeric(df):
    """Ensures all game columns are strictly numeric, fixing any dirty string spacing."""
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['status', 'Group_Status', 'ID']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(' ', ''), errors='coerce')
    return df

def generate_distribution_plots():
    print("Loading datasets for distribution analysis...")
    df_elderly = pd.read_csv('Eldery_Final_ML_Ready.csv')
    df_children = pd.read_csv('Children_Final_ML_Ready.csv')

    df_elderly = coerce_numeric(df_elderly)
    df_children = coerce_numeric(df_children)

    # Automatically identify the target column
    target_col_elderly = 'Group_Status' if 'Group_Status' in df_elderly.columns else 'status'
    target_col_children = 'Group_Status' if 'Group_Status' in df_children.columns else 'status'

    # ==========================================
    # 1. Boxplot: Reaction Time by Status & Gender (Elderly)
    # ==========================================
    plt.figure(figsize=(14, 8))
    
    # Let's find a valid reaction time column to plot (prefer T1 if it exists)
    rt_cols = [c for c in df_elderly.columns if 'reaction_time' in c.lower() and 'T1_' in c]
    if not rt_cols:
        rt_cols = [c for c in df_elderly.columns if 'reaction_time' in c.lower()]

    if rt_cols:
        plot_var = rt_cols[0]
        sns.boxplot(data=df_elderly, x=target_col_elderly, y=plot_var, hue='Gender', palette='Set2')
        plt.title(f'Distribution of {plot_var} by Status and Gender (Elderly)', fontsize=16)
        plt.xlabel('Clinical Status', fontsize=12)
        plt.ylabel('Reaction Time (seconds)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('Elderly_ReactionTime_Boxplot.png', dpi=300)
        print(f"Saved Elderly_ReactionTime_Boxplot.png for variable {plot_var}")
        plt.close()

    # ==========================================
    # 2. Boxplot: Flight Time by Tablet Frequency (Elderly)
    # ==========================================
    plt.figure(figsize=(14, 8))
    
    # Find a valid flight time column (prefer T12 dragging test if it exists)
    ft_cols = [c for c in df_elderly.columns if 'flight_time' in c.lower() and 'T12_' in c]
    if not ft_cols:
        ft_cols = [c for c in df_elderly.columns if 'flight_time' in c.lower()]

    if ft_cols and 'Frequency_Tablet_Use' in df_elderly.columns:
        plot_var = ft_cols[0]
        # Convert to string so categorical palette applies cleanly without assuming it's a continuous number
        df_elderly['Freq_Use_Str'] = df_elderly['Frequency_Tablet_Use'].astype(str)
        
        sns.boxplot(data=df_elderly, x=target_col_elderly, y=plot_var, hue='Freq_Use_Str', palette='coolwarm')
        plt.title(f'Distribution of {plot_var} by Status and Tablet Use Frequency', fontsize=16)
        plt.xlabel('Clinical Status', fontsize=12)
        plt.ylabel('Flight Time', fontsize=12)
        plt.legend(title='Tablet Use Frequency\n(-1 = Unknown)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('Elderly_FlightTime_Boxplot.png', dpi=300)
        print(f"Saved Elderly_FlightTime_Boxplot.png for variable {plot_var}")
        plt.close()

    # ==========================================
    # 3. Histogram: Age Distribution (Children)
    # ==========================================
    plt.figure(figsize=(12, 7))
    if 'Age' in df_children.columns:
        sns.histplot(data=df_children, x='Age', hue=target_col_children, multiple="stack", palette='tab10', kde=True)
        plt.title('Age Distribution by Clinical Status (Children)', fontsize=16)
        plt.xlabel('Age (Years)', fontsize=12)
        plt.ylabel('Patient Count', fontsize=12)
        plt.tight_layout()
        plt.savefig('Children_Age_Distribution.png', dpi=300)
        print("Saved Children_Age_Distribution.png")
        plt.close()

if __name__ == "__main__":
    generate_distribution_plots()