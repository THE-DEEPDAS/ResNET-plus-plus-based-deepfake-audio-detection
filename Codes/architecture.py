import matplotlib.pyplot as plt
import networkx as nx

def create_model_architecture_diagram():
    # Create a directed graph
    G = nx.DiGraph()

    # Define nodes with labels
    nodes = [
        "Input Layer\n(3x224x224)",
        "ResNet50\nBackbone",
        "CBAM Attention\nModule",
        "SE Blocks",
        "Transformer\nBlocks",
        "Multi-Scale\nFeature Fusion",
        "Residual\nConnections",
        "Global Avg\nPooling",
        "Fully Connected\nLayers",
        "Softmax\nClassifier"
    ]

    # Add edges to represent flow
    edges = [
        (nodes[0], nodes[1]),
        (nodes[1], nodes[2]),
        (nodes[2], nodes[3]),
        (nodes[3], nodes[4]),
        (nodes[4], nodes[5]),
        (nodes[5], nodes[6]),
        (nodes[6], nodes[7]),
        (nodes[7], nodes[8]),
        (nodes[8], nodes[9])
    ]

    G.add_edges_from(edges)

    # Create figure with white background
    plt.figure(figsize=(15, 6), facecolor='white')
    plt.gcf().patch.set_facecolor('white')

    # Define custom positions for a top-down flow of nodes
    pos = {
        "Input Layer\n(3x224x224)": (0, 9),
        "ResNet50\nBackbone": (0, 8),
        "CBAM Attention\nModule": (0, 7),
        "SE Blocks": (0, 6),
        "Transformer\nBlocks": (0, 5),
        "Multi-Scale\nFeature Fusion": (0, 4),
        "Residual\nConnections": (0, 3),
        "Global Avg\nPooling": (0, 2),
        "Fully Connected\nLayers": (0, 1),
        "Softmax\nClassifier": (0, 0)
    }

    # Draw the graph
    nx.draw(
        G, 
        pos=pos, 
        with_labels=True, 
        node_color='lightsteelblue', 
        edge_color='darkgrey',
        node_size=3000, 
        font_size=8, 
        font_weight='bold',
        edgecolors='black',
        linewidths=1,
        arrows=True,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.2'
    )

    plt.title('ResNetPlusPlus Model Architecture', color='darkblue', fontsize=15, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Call the function to generate the diagram
create_model_architecture_diagram()