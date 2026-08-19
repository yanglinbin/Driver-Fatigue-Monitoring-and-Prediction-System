import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# Create canvas
fig, ax = plt.subplots(figsize=(10, 12))

# Define original ResNet18 architecture
modules = [
    "Input Layer (3, 224, 224)",
    "7×7 Conv, 64 channels, stride=2",
    "Batch Normalization",
    "ReLU",
    "3×3 Max Pooling, stride=2",
    "Residual Block 1-1 (64 channels)",
    "Residual Block 1-2 (64 channels)",
    "Residual Block 2-1 (128 channels, downsample)",
    "Residual Block 2-2 (128 channels)",
    "Residual Block 3-1 (256 channels, downsample)",
    "Residual Block 3-2 (256 channels)",
    "Residual Block 4-1 (512 channels, downsample)",
    "Residual Block 4-2 (512 channels)",
    "Global Average Pooling",
    "Fully Connected (1000)",
    "Output: 1000 classes"
]

# Set module colors
colors = {
    'Conv': 'lightblue',
    'Normalization': 'lightsalmon',
    'Pooling': 'lightgreen',
    'Residual': 'lightyellow',
    'ReLU': 'lightgray',
    'Fully': 'pink',
    'Input': 'white',
    'Output': 'white'
}

# Expand the diagram to show the structure of a basic residual block
residual_blocks = [5, 6, 7, 8, 9, 10, 11, 12]  # Indices of residual blocks

# Draw modules
height = 0.6
width = 0.7
for i, module in enumerate(modules):
    # Determine module color
    color = 'white'
    for key, val in colors.items():
        if key in module:
            color = val
            break

    # Draw module box
    x_position = 0.15
    box_width = width

    # Draw residual connection for residual blocks
    if i in residual_blocks:
        skip_y = 15 - i + height / 2
        skip_arrow = FancyArrowPatch(
            (x_position + width / 2, skip_y - height),
            (x_position + width / 2, skip_y + height),
            arrowstyle='-', linewidth=1.5, linestyle='--', color='red'
        )
        ax.add_patch(skip_arrow)

    rect = Rectangle((x_position, 15 - i), box_width, height, facecolor=color, edgecolor='black', alpha=0.8)
    ax.add_patch(rect)
    ax.text(x_position + box_width / 2, 15 - i + height / 2, module, ha='center', va='center', fontsize=10)

    # Add connecting arrows
    if i < len(modules) - 1:
        arrow = FancyArrowPatch(
            (x_position + box_width / 2, 15 - i),
            (x_position + box_width / 2, 15 - i - 0.2),
            arrowstyle='->', linewidth=1.5, mutation_scale=15
        )
        ax.add_patch(arrow)

# Add annotation for the residual connections
ax.text(0.9, 8.5, "Residual\nConnections", ha='center', va='center', color='red', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7, edgecolor='red'))

# Add explanation of basic block
ax.text(0.05, 2,
        "Each Residual Block contains:\n- 3×3 Conv\n- Batch Norm\n- ReLU\n- 3×3 Conv\n- Batch Norm\n+ Identity/Projection Shortcut\n- ReLU",
        fontsize=9, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))

# Set figure parameters
ax.set_xlim(0, 1)
ax.set_ylim(0, 16)
ax.axis('off')
plt.title("Original ResNet18 Model Architecture")
plt.tight_layout()

# Save figures
plt.savefig("original_resnet18_architecture.pdf", bbox_inches='tight', dpi=300)
plt.savefig("original_resnet18_architecture.png", bbox_inches='tight', dpi=300)
plt.show()

print("Original ResNet18 architecture diagrams generated: original_resnet18_architecture.pdf/png")