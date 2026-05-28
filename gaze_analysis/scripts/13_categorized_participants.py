import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
RESULTS_DIR = "gaze_analysis/results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Load the demographic data
file_path = 'gaze_analysis/data/processed/demographicsv2.csv'
df = pd.read_csv(file_path)

# 1. Categorize by Age (Old > 50, Young <= 50)
df['Age_Category'] = df['Age'].apply(lambda x: 'Old' if x > 50 else 'Young')

# 2. Define colors to highlight those without tablets/smartphones
# Red is used for 'No' to clearly identify them in the plot
palette = {'Yes': '#2ecc71', 'No': '#e74c3c'}

# Sort data for a structured visualization
df_sorted = df.sort_values(by=['Age_Category', 'Age'], ascending=[False, True])

# 3. Create the Visualization
plt.figure(figsize=(10, 12))
sns.set_theme(style="whitegrid")

# Bar plot showing names (Code), Age, and device ownership
ax = sns.barplot(
    data=df_sorted,
    x='Age',
    y='Code',
    hue='Has Tablet ot Smart Phone',
    palette=palette,
    dodge=False
)

# Formatting and labels
plt.title('Participant Demographics: Age and Device Ownership\n(Red = No Tablet/Smartphone)', fontsize=16, fontweight='bold')
plt.xlabel('Age (Years)', fontsize=12)
plt.ylabel('Participant Name (Code)', fontsize=12)
plt.legend(title='Has Tablet or Smart Phone', loc='lower right')

# Vertical line at 50 to visually separate categories
plt.axvline(x=50, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()

# Save the plot directly to your results folder
plot_output = os.path.join(RESULTS_DIR, 'Participant_Age_Device_Visualization.png')
plt.savefig(plot_output, dpi=300)
plt.close()

# Save the categorized list for reference
csv_output = os.path.join(RESULTS_DIR, 'Categorized_Participants.csv')
df_sorted[['Code', 'Age', 'Age_Category', 'Has Tablet ot Smart Phone']].to_csv(csv_output, index=False)

print(f"✅ Saved plot to: {plot_output}")
print(f"✅ Saved CSV to: {csv_output}")