import matplotlib.pyplot as plt
import numpy as np

# Define epochs
epochs = np.array([10, 15, 20, 25, 28, 29, 30, 32, 35, 38, 40])

# New accuracy and loss arrays
accuracy = np.array([
    55, 74, 91, 93, 93.7, 93.9, 92.8, 93.8, 94.1, 94.3, 94.45
])

loss = np.array([
    4.5, 2.4, 1.2, 0.5, 0.3, 0.25, 0.22, 0.149, 0.103, 0.043, 0.0172
])

# Plot validation accuracy vs. epochs
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, marker='', label="Validation Accuracy", color="blue")
plt.title("Validation Accuracy vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()

# Plot validation loss vs. epochs
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, marker='', label="Validation Loss", color="red")
plt.title("Validation Loss vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
