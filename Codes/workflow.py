import matplotlib.pyplot as plt
import networkx as nx

def create_methodology_workflow_diagram():
    # Create a directed graph
    G = nx.DiGraph()

    # Define workflow stages with nodes
    preprocessing = [
        "Raw Audio Input",
        "Normalization",
        "Noise Reduction",
        "Mel Spectrogram\nExtraction",
        "Histogram\nEqualization"
    ]

    feature_enhancement = [
        "Pre-Emphasis\nFiltering",
        "Delta Features\nComputation",
        "Voice Activity\nDetection"
    ]

    data_preparation = [
        "Protocol File\nMapping",
        "Data\nAugmentation"
    ]

    model_architecture = [
        "ResNet50\nBackbone",
        "Attention\nMechanisms",
        "Multi-Scale\nFeature Fusion"
    ]

    training_pipeline = [
        "Model Training",
        "Validation",
        "Performance\nEvaluation"
    ]

    # Create connections between stages
    workflow_stages = preprocessing + feature_enhancement + \
                      data_preparation + model_architecture + training_pipeline

    edges = [
        (preprocessing[i], preprocessing[i+1]) for i in range(len(preprocessing)-1)
    ] + [
        (preprocessing[-1], feature_enhancement[0])
    ] + [
        (feature_enhancement[i], feature_enhancement[i+1]) for i in range(len(feature_enhancement)-1)
    ] + [
        (feature_enhancement[-1], data_preparation[0])
    ] + [
        (data_preparation[0], data_preparation[1])
    ] + [
        (data_preparation[1], model_architecture[0])
    ] + [
        (model_architecture[i], model_architecture[i+1]) for i in range(len(model_architecture)-1)
    ] + [
        (model_architecture[-1], training_pipeline[0])
    ] + [
        (training_pipeline[i], training_pipeline[i+1]) for i in range(len(training_pipeline)-1)
    ]

    G.add_edges_from(edges)

    # Create figure with white background
    plt.figure(figsize=(15, 10), facecolor='white')
    plt.gcf().patch.set_facecolor('white')

    # Remove or comment out the line that uses spring_layout:
    # pos = nx.spring_layout(G, k=0.5, iterations=50)

    pos = {
        "Raw Audio Input": (0, 7),
        "Normalization": (0, 6),
        "Noise Reduction": (0, 5),
        "Mel Spectrogram\nExtraction": (0, 4),
        "Histogram\nEqualization": (0, 3),
        "Pre-Emphasis\nFiltering": (0, 2),
        "Delta Features\nComputation": (0, 1),
        "Voice Activity\nDetection": (0, 0),
        "Protocol File\nMapping": (1, 0),
        "Data\nAugmentation": (2, 0),
        "ResNet50\nBackbone": (3, 0),
        "Attention\nMechanisms": (4, 0),
        "Multi-Scale\nFeature Fusion": (5, 0),
        "Model Training": (6, 1),
        "Validation": (6, 0),
        "Performance\nEvaluation": (6, -1),
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

    plt.title('Audio Spoofing Detection Methodology Workflow', color='darkblue', fontsize=15, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Call the function to generate the diagram
create_methodology_workflow_diagram()