import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import textwrap

def generate_cohort_infographic():
    print("Generating Clinical Cohort Profiles Infographic...")
    
    # 1. Define the Cohort Data
    cohorts = [
        {
            "name": "Parkinson's Disease",
            "color": "#e74c3c", # Red
            "traits": [
                "• Notably shortened reaction times",
                "• Prolonged touch times on the screen",
                "• Severe reduction in complex tapping & dragging"
            ]
        },
        {
            "name": "Falls & Cog. Decline",
            "color": "#e67e22", # Orange
            "traits": [
                "• Significantly prolonged touch times",
                "• Sharp decrease in flight times (heavy resting)",
                "• Extreme shift toward medium-pressure application"
            ]
        },
        {
            "name": "Diabetes",
            "color": "#f1c40f", # Yellow
            "traits": [
                "• Generally faster interaction speeds (decreased touch and flight times)",
                "• Erratic kinematics (highly elevated tap counts)"
            ]
        },
        {
            "name": "IDD",
            "color": "#2ecc71", # Green
            "traits": [
                "• Severely prolonged touch and test times",
                "• Extreme reliance on medium pressure",
                "• Dramatically elevated error rates (touch outside)"
            ]
        },
        {
            "name": "SMA (Pediatric)",
            "color": "#3498db", # Blue
            "traits": [
                "• Prolonged flight and touch times",
                "• Distinct shift away from high-pressure interactions",
                "• General reduction in complex kinematics"
            ]
        },
        {
            "name": "Chronic Otitis Media",
            "color": "#9b59b6", # Purple
            "traits": [
                "• Prolonged overall test and interaction times",
                "• Significantly elevated error rates (clumsiness)",
                "• Increased number of accidental drag attempts"
            ]
        }
    ]

    # 2. Setup the Figure
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off') # Hide the grid and axes
    
    # Title
    plt.text(50, 95, "Digital Phenotypes: Patient Cohort Characteristics", 
             ha='center', va='center', fontsize=22, fontweight='bold', color='#2c3e50')
    plt.text(50, 91, "Primary biomechanical and interactive difficulties derived from targeted tablet assessments", 
             ha='center', va='center', fontsize=14, fontstyle='italic', color='#7f8c8d')

    # 3. Grid Layout Math (2 Columns, 3 Rows)
    card_width = 42
    card_height = 20
    x_starts = [6, 52] # X coordinates for Column 1 and Column 2
    y_starts = [65, 40, 15] # Y coordinates for Row 1, 2, 3
    
    # 4. Draw the Cards
    for i, cohort in enumerate(cohorts):
        col = i % 2
        row = i // 2
        
        x = x_starts[col]
        y = y_starts[row]
        
        # Draw Card Background (Light Grey with soft edge)
        bg_box = mpatches.FancyBboxPatch((x, y), card_width, card_height, 
                                         boxstyle="round,pad=0.5,rounding_size=1.5", 
                                         ec="none", fc="#f8f9fa", zorder=1)
        ax.add_patch(bg_box)
        
        # Draw Card Header Banner (Color Coded)
        header_box = mpatches.FancyBboxPatch((x, y + card_height - 4), card_width, 4, 
                                             boxstyle="round,pad=0.5,rounding_size=1.5", 
                                             ec="none", fc=cohort["color"], zorder=2)
        # To make the header flat on the bottom, we just draw a rectangle over the bottom rounded corners
        flat_bottom = mpatches.Rectangle((x, y + card_height - 4), card_width, 2, fc=cohort["color"], zorder=2)
        ax.add_patch(header_box)
        ax.add_patch(flat_bottom)
        
        # Add Header Text (Status Name)
        plt.text(x + card_width/2, y + card_height - 2, cohort["name"].upper(), 
                 ha='center', va='center', fontsize=14, fontweight='bold', color='white', zorder=3)
        
        # Add Bullet Points (Wrapped to fit the card)
        text_y = y + card_height - 8
        for trait in cohort["traits"]:
            # Wrap text so it doesn't spill out of the box
            wrapped_text = textwrap.fill(trait, width=45)
            plt.text(x + 2, text_y, wrapped_text, ha='left', va='top', 
                     fontsize=12, color='#34495e', zorder=3, linespacing=1.4)
            
            # Move down for the next bullet point based on how many lines this one took
            line_count = len(wrapped_text.split('\n'))
            text_y -= (line_count * 2.5)

    # 5. Save the Image
    output_filename = "Clinical_Profiles_Infographic.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Success! Saved beautiful infographic to '{output_filename}'")

if __name__ == "__main__":
    generate_cohort_infographic()
    