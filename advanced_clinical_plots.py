import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import warnings

warnings.filterwarnings('ignore')

def coerce_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['status', 'Group_Status', 'ID']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(' ', ''), errors='coerce')
    return df

def plot_deviation_heatmap(csv_file, top_features, dataset_name, healthy_label='healthy'):
    """Generates a heatmap showing % difference from the healthy baseline."""
    print(f"\n--- Generating Deviation Heatmap for {dataset_name} ---")
    
    df = pd.read_csv(csv_file)
    df = coerce_numeric(df)
    target_col = 'status' if 'status' in df.columns else 'Group_Status'
    
    actual_healthy = next((s for s in df[target_col].dropna().unique() if healthy_label.lower() in s.lower()), None)
    if not actual_healthy:
        print("Error: Could not find healthy group.")
        return

    # Calculate Medians
    medians = df.groupby(target_col)[top_features].median()
    
    # Calculate % Deviation from Healthy
    healthy_vals = medians.loc[actual_healthy]
    deviation_df = ((medians - healthy_vals) / healthy_vals) * 100
    
    # Drop the healthy row (since it will just be all 0%)
    deviation_df = deviation_df.drop(index=actual_healthy)
    
    # Clean up column names for the plot
    deviation_df.columns = [c.replace('_DOM_', '\n').replace('_', ' ') for c in deviation_df.columns]
    
    plt.figure(figsize=(12, 6))
    # Using a coolwarm colormap: Red = Higher/Slower/More Errors, Blue = Lower/Faster
    sns.heatmap(deviation_df, annot=True, fmt=".0f", cmap="coolwarm", center=0, 
                cbar_kws={'label': '% Change vs Healthy Baseline'}, linewidths=1)
    
    plt.title(f"{dataset_name}: Clinical Deviation from Healthy Baseline (%)", fontsize=16, pad=15)
    plt.ylabel("Clinical Status", fontsize=12)
    plt.xlabel("Tablet Features", fontsize=12)
    plt.xticks(rotation=0)
    
    filename = f"{dataset_name}_Deviation_Heatmap.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved Heatmap to '{filename}'")


def plot_feature_violins(csv_file, top_features, dataset_name):
    print(f"\n--- Generating Violin Plots for {dataset_name} ---")

    # --------------------------------------------------
    # 1. Load Data
    # --------------------------------------------------
    df = pd.read_csv(csv_file)

    target_col = 'status' if 'status' in df.columns else 'Group_Status'

    # Clean status labels early
    df[target_col] = df[target_col].astype(str).str.strip().str.title()

    # --------------------------------------------------
    # 2. Convert features to numeric
    # --------------------------------------------------
    for col in top_features:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(' ', ''),
                errors='coerce'
            )

    # --------------------------------------------------
    # 3. Keep only required columns
    # --------------------------------------------------
    plot_df = df[[target_col] + top_features]

    # --------------------------------------------------
    # 4. Melt dataframe for seaborn
    # --------------------------------------------------
    melted_df = plot_df.melt(
        id_vars=target_col,
        value_vars=top_features,
        var_name='Feature',
        value_name='Score'
    )

    # Remove NaNs
    melted_df = melted_df.dropna(subset=['Score'])

    # --------------------------------------------------
    # 5. Remove Parkinson artifacts (Elderly dataset)
    # --------------------------------------------------
    if dataset_name == "Elderly":

        rogue_mask = (
            melted_df[target_col].str.contains('Parkinson', case=False, na=False)
            & melted_df['Feature'].str.match(r'^(T1_|T2_|T12_)')
        )

        melted_df = melted_df[~rogue_mask]

    # --------------------------------------------------
    # 6. Clean feature names for plotting
    # --------------------------------------------------
    melted_df['Feature'] = melted_df['Feature'].apply(
        lambda x: x.replace('_DOM_', '\n').replace('_', ' ')
    )

    # --------------------------------------------------
    # 7. Extract status order from actual plotted data
    # --------------------------------------------------
    unique_statuses = sorted(melted_df[target_col].dropna().unique())

    # Optional clinical ordering (if present)
    if dataset_name == "Elderly":
        preferred_order = [
        "Healthy",
        "Parkinson",
        "Idd",
        "Diabetes",
        "Falls And Cognitive Decline"
        ]
    else:  # Children
        preferred_order = [
            "Healthy",
            "Sma",
            "Chronic Otitis Media"
        ]
    ordered_statuses = [s for s in preferred_order if s in unique_statuses]

    if ordered_statuses:
        unique_statuses = ordered_statuses

    # --------------------------------------------------
    # 8. Plot violin charts (with colors per status)
    # --------------------------------------------------

    # Create a color palette for the statuses
    palette_colors = sns.color_palette("Set2", n_colors=len(unique_statuses))
    status_palette = dict(zip(unique_statuses, palette_colors))

    g = sns.catplot(
        data=melted_df,
        x=target_col,
        y='Score',
        hue=target_col,                # <-- enables coloring
        col='Feature',
        col_wrap=3,
        kind='violin',
        sharey=False,
        height=4,
        aspect=1.2,
        inner="quartile",
        order=unique_statuses,
        hue_order=unique_statuses,
        palette=status_palette,        # <-- apply colors
        dodge=False                    # keeps violins centered
    )

    # --------------------------------------------------
    # 9. Formatting
    # --------------------------------------------------
    g.fig.suptitle(
        f"{dataset_name}: Patient Distributions Across Top Tablet Features",
        fontsize=18,
        y=1.05
    )

    g.set_titles("{col_name}", size=12, fontweight='bold')
    g.set_axis_labels("", "Raw Score")

    # Rotate status labels for readability
    for ax in g.axes.flatten():
        ax.tick_params(axis='x', rotation=30)

    # --------------------------------------------------
    # 10. Save Figure
    # --------------------------------------------------
    filename = f"{dataset_name}_Feature_Violins.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"Saved Violin Plots to '{filename}'")

    plt.show()

if __name__ == "__main__":
    # The raw dataset file (NOT the medians file, we need raw data for violins)
    elderly_file = 'data/Eldery_Final_ML_Ready.csv'
    
    # Pick the 3 to 6 best features that defined your XGBoost models
    top_elderly_features = [
        "T2_DOM_Touch_Time",      
        "T1_DOM_medium_pressure",  
        "T2_DOM_reaction_time",     
        "T1_DOM_Flight_Time",
        "T2_DOM_Number_Taps",
        "T20_DOM_Reaction_Time" 
    ]
    
    plot_deviation_heatmap(elderly_file, top_elderly_features, "Elderly")
    plot_feature_violins(elderly_file, top_elderly_features, "Elderly")

        # The raw dataset file (NOT the medians file, we need raw data for violins)
    children_file = 'data/Children_Final_ML_Ready.csv'
    
    # Pick the 3 to 6 best features that defined your XGBoost models
    top_children_features = [
        "T1_DOM_high_pressure",      
        "T5_DOM_Test_Duration",  
        "T20_DOM_Medium_Pressure",     
        "T2_DOM_low_pressure",
        "T1_DOM_Touch_Time",
        "T12_DOM_Drag_Attempts" 
    ]
    
    plot_deviation_heatmap(children_file, top_children_features, "Children")
    plot_feature_violins(children_file, top_children_features, "Children")