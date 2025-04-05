import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define the boxes and arrows
    boxes = [
        {"xy": (0.1, 0.8), "width": 0.3, "height": 0.1, "text": "Load Audio Data"},
        {"xy": (0.1, 0.6), "width": 0.3, "height": 0.1, "text": "Preprocess Audio"},
        {"xy": (0.1, 0.4), "width": 0.3, "height": 0.1, "text": "Extract Features"},
        {"xy": (0.1, 0.2), "width": 0.3, "height": 0.1, "text": "Create Dataset"},
        {"xy": (0.6, 0.8), "width": 0.3, "height": 0.1, "text": "Define Model"},
        {"xy": (0.6, 0.6), "width": 0.3, "height": 0.1, "text": "Train Model"},
        {"xy": (0.6, 0.4), "width": 0.3, "height": 0.1, "text": "Evaluate Model"},
        {"xy": (0.6, 0.2), "width": 0.3, "height": 0.1, "text": "Save Best Model"},
    ]

    arrows = [
        {"xy": (0.25, 0.8), "xytext": (0.25, 0.7)},
        {"xy": (0.25, 0.6), "xytext": (0.25, 0.5)},
        {"xy": (0.25, 0.4), "xytext": (0.25, 0.3)},
        {"xy": (0.75, 0.8), "xytext": (0.75, 0.7)},
        {"xy": (0.75, 0.6), "xytext": (0.75, 0.5)},
        {"xy": (0.75, 0.4), "xytext": (0.75, 0.3)},
        {"xy": (0.4, 0.25), "xytext": (0.6, 0.25)},
    ]

    # Draw the boxes
    for box in boxes:
        rect = patches.FancyBboxPatch(box["xy"], box["width"], box["height"],
                                      boxstyle="round,pad=0.1", edgecolor="black", facecolor="lightblue")
        ax.add_patch(rect)
        rx, ry = rect.get_x(), rect.get_y()
        cx = rx + rect.get_width() / 2.0
        cy = ry + rect.get_height() / 2.0
        ax.annotate(box["text"], (cx, cy), color='black', weight='bold',
                    fontsize=10, ha='center', va='center')

    # Draw the arrows
    for arrow in arrows:
        ax.annotate("", xy=arrow["xy"], xytext=arrow["xytext"],
                    arrowprops=dict(arrowstyle="->", color='black'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title('Approach Flowchart')
    plt.show()

if __name__ == "__main__":
    draw_flowchart()
