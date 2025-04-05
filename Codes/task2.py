import cv2
import numpy as np
import hashlib
import matplotlib.pyplot as plt
from collections import defaultdict

image_path = './copy_move_image.png'  # Change this to your image file path
image = cv2.imread(image_path)

# Step 2: Convert to Grayscale (if it's a color image)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply DCT (Discrete Cosine Transform)
dct_image = cv2.dct(np.float32(gray_image))  # Perform 2D DCT

# Step 4: Sliding Window and Hash Extraction
window_height = 8  # Height of the block (same as the region size)
window_width = 5   # Width of the block (same as the region size)

# Step 5: Create a dictionary to track the frequency of regions and their positions
region_hash_map = defaultdict(list)  # A dictionary to store positions for each hash

# Sliding window approach to check all regions
height, width = gray_image.shape

for i in range(height - window_height + 1):  # Loop through the image with a sliding window
    for j in range(width - window_width + 1):
        # Extract the current window's DCT coefficients
        window = dct_image[i:i + window_height, j:j + window_width]
        
        # Hash the current window
        window_hash = hashlib.md5(window.flatten().tobytes()).hexdigest()
        
        # Store the position of the matching window in the dictionary
        region_hash_map[window_hash].append((i, j))

# Step 6: Identify most frequent forged regions with at least 2 occurrences
forged_regions_count = 0
forged_regions_positions = []

# Filter out hash entries that appear less than 2 times
for region_hash, positions in region_hash_map.items():
    if len(positions) > 1:  # If a region appears more than once
        forged_regions_count += 1
        forged_regions_positions.extend(positions)  # Add all positions of the forged region

# Step 7: Highlight the regions with maximum overlap (yellow)
highlighted_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for highlighting

# Highlight all forged regions in yellow
for (i, j) in forged_regions_positions:
    highlighted_image[i:i + window_height, j:j + window_width] = [0, 255, 255]  # Yellow color

# Step 8: Display the image with highlighted regions in yellow using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
plt.title(f"Detected {forged_regions_count} Forged Region(s)")
plt.axis('off')
plt.show()

# Step 9: Print the positions of the forged regions in the terminal
if forged_regions_count > 0:
    print(f"Found {forged_regions_count} forged regions:")
    for pos in forged_regions_positions:
        print(f"Position: {pos}")
else:
    print("No forged regions detected.")
