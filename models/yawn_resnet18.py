import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# Create canvas
fig, ax = plt.subplots(figsize=(8, 10))

# Define Yawn ResNet18 main modules (simplified version)
modules = [
    "Input Layer (3, 224, 224)",
    "7×7 Conv, 64 channels",
    "Max Pooling",
    "Residual Block 1 (64 channels)",
    "Residual Block 2 (128 channels)",
    "Residual Block 3 (256 channels)",
    "Residual Block 4 (512 channels)",
    "Global Average Pooling",
    "Dropout(0.5)",
    "Fully Connected (2)",
    "Output: Yawn/No Yawn"
]

# Set module colors
colors = {
    'Conv': 'lightblue',
    'Pooling': 'lightgreen',
    'Residual': 'lightyellow',
    'Dropout': 'lightpink',
    'Fully': 'lightsalmon',
    'ReLU': 'lightgray',
    'Input': 'white',
    'Output': 'white'
}

# Draw modules
height = 0.6
width = 0.6
for i, module in enumerate(modules):
    # Determine module color
    color = 'white'
    for key, val in colors.items():
        if key in module:
            color = val
            break

    # Draw module box
    rect = Rectangle((0.2, 10 - i), width, height, facecolor=color, edgecolor='black', alpha=0.8)
    ax.add_patch(rect)
    ax.text(0.2 + width / 2, 10 - i + height / 2, module, ha='center', va='center', fontsize=10)

    # Add connecting arrows
    if i < len(modules) - 1:
        arrow = FancyArrowPatch((0.2 + width / 2, 10 - i), (0.2 + width / 2, 10 - i - 0.2),
                                arrowstyle='->', linewidth=1.5, mutation_scale=15)
        ax.add_patch(arrow)

# Set figure parameters
ax.set_xlim(0, 1)
ax.set_ylim(0, 11)
ax.axis('off')
plt.title("Yawn Detection ResNet18 Model Architecture")
plt.tight_layout()

# Save figures
plt.savefig("yawn_resnet_model_simplified.pdf", bbox_inches='tight', dpi=300)
plt.savefig("yawn_resnet_model_simplified.png", bbox_inches='tight', dpi=300)
plt.show()

print("Simplified model architecture diagrams generated: yawn_resnet_model_simplified.pdf/png")