import matplotlib.pyplot as plt

# Metrics and their values
metrics = [
    'Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy', 
    'Precision', 'Recall', 'F1 Score', 'FPR', 'FNR', 'TPR', 'TNR'
]
values = [
    0.0096, 99.74, 0.3425, 89.82, 
    97.56, 79.08, 87.35, 1.58, 20.92, 79.08, 98.42
]

# Create the bar plot
plt.figure(figsize=(12, 6))
plt.bar(metrics, values, color='skyblue', edgecolor='black')

# Add labels, title, and rotate x-axis labels
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Values (%)', fontsize=12)
plt.title('Performance Metrics Overview', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Show the plot
plt.tight_layout()
plt.show()
