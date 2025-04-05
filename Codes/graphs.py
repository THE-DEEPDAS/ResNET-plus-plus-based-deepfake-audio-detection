import matplotlib.pyplot as plt

# Data
epochs = [10, 15]
accuracy = [50.12, 89.82]

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, accuracy, marker='o', linestyle='-', color='b', label='Accuracy')

# Add labels and title
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Accuracy vs Epochs', fontsize=14)

# Add grid and legend
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)

# Show the plot
plt.tight_layout()
plt.show()
