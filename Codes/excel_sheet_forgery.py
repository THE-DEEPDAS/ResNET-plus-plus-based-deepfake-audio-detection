import pandas as pd
import hashlib

# Load data from Excel file
def load_data_from_excel(file_path):
    # Read Excel file into a DataFrame
    df = pd.read_excel(file_path, header=None)
    # Convert DataFrame to a list of lists (matrix form)
    data = df.values.tolist()
    return data

# Generate a unique hash for an 8x5 block starting at (r, c)
def hash_block(data, r, c):
    # Create a hash object
    hash_obj = hashlib.md5()
    # Hash each element in the 8x5 block to generate a unique hash for the block
    for i in range(8):
        for j in range(5):
           hash_obj.update(str(data[r + i][c + j]).encode('utf-8'))
    return hash_obj.hexdigest()

# Find and print similar 8x5 blocks in the matrix
def find_similar_blocks(data):
    rows = len(data)
    cols = len(data[0]) if rows > 0 else 0
    hash_map = {}  # Dictionary to store block hash and its top-left position (row, col)
    found = False

    # Traverse all possible 8x5 blocks
    for r in range(rows - 7):  # Limit to rows that can start an 8x5 block
        for c in range(cols - 4):  # Limit to columns that can start an 8x5 block
            block_hash = hash_block(data, r, c)

            # Check if this block hash already exists in the map
            if block_hash in hash_map:
                # Get the previous position of the matching block
                prev_r, prev_c = hash_map[block_hash]
                found = True
                print("Similar block found:")
                print(f"Block 1: Rows {prev_r} to {prev_r + 7}, Columns {prev_c} to {prev_c + 4}")
                print(f"Block 2: Rows {r} to {r + 7}, Columns {c} to {c + 4}")
                print()
            else:
                # Store the hash and current top-left position if it's unique
                hash_map[block_hash] = (r, c)

    if not found:
        print("No similar 8x5 blocks found.")

# Main function to execute the code
def main():
    file_path = 'kem.xlsx'  # Replace with your actual file path
    data = load_data_from_excel(file_path)
    find_similar_blocks(data)

if __name__ == "__main__":
    main()
