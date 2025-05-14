import pandas as pd
import numpy as np
import json

# Step 1: Load the CSV file
df = pd.read_csv('common.csv')

# Step 2: Get unique XIDs and shuffle them
unique_xids = df['XID'].unique().tolist()
np.random.seed(42)  # for reproducibility
np.random.shuffle(unique_xids)

# Step 3: Calculate split sizes based on 8:1:1 ratio
total_xids = len(unique_xids)
train_size = int(0.8 * total_xids)
val_size = int(0.1 * total_xids)

# Step 4: Assign XIDs to splits
train_xids = unique_xids[:train_size]
val_xids = unique_xids[train_size:train_size+val_size]
test_xids = unique_xids[train_size+val_size:]

# Step 5: Create a dictionary of splits
split_ids = {
    'train': train_xids,
    'val': val_xids,
    'test': test_xids
}

# Step 6: Save the split IDs to a JSON file
with open('dataset_split_ids.json', 'w') as f:
    json.dump(split_ids, f)

print(f"Split IDs saved to dataset_split_ids.json")
print(f"Train: {len(train_xids)} XIDs")
print(f"Validation: {len(val_xids)} XIDs")
print(f"Test: {len(test_xids)} XIDs")