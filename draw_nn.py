import matplotlib.pyplot as plt
import numpy as np

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    """
    Draw a neural network cartoon using matplotlib.
    
    Args:
        ax: matplotlib axes
        left, right, bottom, top: boundaries of the plot
        layer_sizes: list of layer sizes (number of neurons per layer)
    """
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            # Add some light blue fill like the user's image
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='#b3d9ff', ec='black', lw=1.5, zorder=4)
            ax.add_artist(circle)
            
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k', alpha=0.3, lw=0.5)
                ax.add_artist(line)

fig = plt.figure(figsize=(10, 8))
ax = fig.gca()
ax.axis('off')

# Smashifix Topology representation (Scaled down for clean visual tracking)
# Layer 1: Input (Concatenated Features) - e.g., 10 nodes for illustration
# Layer 2: Hidden Layer 1 (Dense) - 12 nodes
# Layer 3: Hidden Layer 2 (Dropout/Reg) - 12 nodes
# Layer 4: Output Layer (6 Stroke Classes) - 6 nodes
layer_sizes = [10, 12, 12, 6]

draw_neural_net(ax, .1, .9, .1, .9, layer_sizes)

# Add Labels
plt.text(0.1, 0.95, 'Vis + Pose Input\n(Fused Features)', horizontalalignment='center', fontsize=12, fontweight='bold')
plt.text(0.36, 0.95, 'Dense Layer 1\n(ReLU)', horizontalalignment='center', fontsize=12, fontweight='bold')
plt.text(0.63, 0.95, 'Dense Layer 2\n(Dropout)', horizontalalignment='center', fontsize=12, fontweight='bold')
plt.text(0.9, 0.95, 'Output Classes\n(Softmax)', horizontalalignment='center', fontsize=12, fontweight='bold')

# Save to Desktop
plt.savefig('smashifix_hybrid_fc.png', dpi=300, bbox_inches='tight')
print("Successfully generated slashifix_hybrid_fc.png")
