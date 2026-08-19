import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# Create canvas
fig, ax = plt.subplots(figsize=(10, 12))

# Define GRU model architecture
modules = [
    "Input Layer (5 features, seq_len=1800)",
    "GRU Layer 1 (hidden_size=64)",
    "Dropout (0.2)",
    "GRU Layer 2 (hidden_size=64)",
    "Select Last Time Step Output",
    "Fully Connected Layer (64 → 1)",
    "Output: Fatigue Score Prediction"
]

# Set module colors
colors = {
    'Input': 'white',
    'GRU': 'lightblue',
    'Dropout': 'lightpink',
    'Select': 'lightgreen',
    'Fully': 'lightsalmon',
    'Output': 'white'
}

# Define GRU cell structure
gru_components = [
    "Update Gate (z)",
    "Reset Gate (r)",
    "Candidate Hidden State (h̃)",
    "Final Hidden State (h)"
]

# Draw modules
height = 0.7
width = 0.7
y_positions = []  # Track y positions to adjust for larger GRU boxes
current_y = 7

for i, module in enumerate(modules):
    # Determine module color
    color = 'white'
    for key, val in colors.items():
        if key in module:
            color = val
            break

    # Draw module box
    x_position = 0.15

    # For GRU layers, make them larger and add internal structure
    if "GRU Layer" in module:
        gru_height = height * 2  # Make GRU boxes taller
        rect = Rectangle((x_position, current_y - gru_height), width, gru_height, facecolor=color, edgecolor='black',
                         alpha=0.8)
        ax.add_patch(rect)

        # Place label above the content
        ax.text(x_position + width / 2, current_y - 0.1, module, ha='center', va='top', fontsize=10, fontweight='bold')

        # Draw GRU cell components with more space
        cell_height = (gru_height - 0.3) / 5  # Reserve space at top for label
        for j, component in enumerate(gru_components):
            cell_y = current_y - gru_height + 0.3 + j * cell_height
            cell_rect = Rectangle((x_position + 0.1, cell_y), width - 0.2, cell_height - 0.05,
                                  facecolor='white', edgecolor='black', alpha=0.7)
            ax.add_patch(cell_rect)
            ax.text(x_position + width / 2, cell_y + cell_height / 2 - 0.025,
                    component, ha='center', va='center', fontsize=8)

        y_positions.append(current_y - gru_height)
        current_y -= (gru_height + 0.3)  # Add extra space after GRU layers
    else:
        rect = Rectangle((x_position, current_y - height), width, height, facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x_position + width / 2, current_y - height / 2, module, ha='center', va='center', fontsize=10)
        y_positions.append(current_y - height)
        current_y -= (height + 0.3)  # Add space between modules

# Add connecting arrows
for i in range(len(modules) - 1):
    arrow = FancyArrowPatch(
        (x_position + width / 2, y_positions[i]),
        (x_position + width / 2, y_positions[i + 1] + height * (2 if "GRU Layer" in modules[i + 1] else 1)),
        arrowstyle='->', linewidth=1.5, mutation_scale=15
    )
    ax.add_patch(arrow)

# Add recurrent connections for GRU layers
for i, module in enumerate(modules):
    if "GRU Layer" in module:
        # Find the y position for this GRU layer
        y_pos = y_positions[i]
        gru_height = height * 2

        # Draw recurrent connection
        h_mid = y_pos + gru_height / 2
        recurrent_arrow = FancyArrowPatch(
            (x_position + width, h_mid),
            (x_position + width + 0.15, h_mid),
            connectionstyle="arc3,rad=0.3",
            arrowstyle='->', linewidth=1.5, mutation_scale=15, color='darkblue'
        )
        ax.add_patch(recurrent_arrow)

        recurrent_arrow2 = FancyArrowPatch(
            (x_position + width + 0.15, h_mid),
            (x_position + width, h_mid),
            connectionstyle="arc3,rad=0.3",
            arrowstyle='->', linewidth=1.5, mutation_scale=15, color='darkblue'
        )
        ax.add_patch(recurrent_arrow2)

        # Add recurrent label
        ax.text(x_position + width + 0.15, h_mid + 0.15, "Recurrent\nConnection",
                ha='center', va='center', fontsize=8, color='darkblue')

# Add explanation
explanation_text = (
    "GRU Model Architecture:\n"
    "- Input features: 5 (Blink Count, Blink Rate,\n  Yawn Count, Yawn Rate, Fatigue Score)\n"
    "- Sequence length: 1800 (30 minutes of data)\n"
    "- Hidden size: 64\n"
    "- Number of GRU layers: 2\n"
    "- Dropout rate: 0.2 (between GRU layers)\n"
    "- Output: Fatigue Score prediction\n"
    "- Training: MSE Loss, Adam optimizer"
)
ax.text(0.05, 1.0, explanation_text, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))

# Set figure parameters
ax.set_xlim(0, 1)
ax.set_ylim(current_y - 0, 7)  # Adjust y-axis limits based on final position
ax.axis('off')
plt.title("Fatigue Prediction GRU Model Architecture")
plt.tight_layout()

# Save figures
plt.savefig("gru_model_architecture.pdf", bbox_inches='tight', dpi=300)
plt.savefig("gru_model_architecture.png", bbox_inches='tight', dpi=300)
plt.show()

print("GRU model architecture diagrams generated: gru_model_architecture.pdf/png")