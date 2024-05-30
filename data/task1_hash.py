import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from scipy.sparse import csr_matrix, vstack
import pickle

# Read the dataset from a CSV file
file_path = 'task1_data.txt'  # Update this path to your actual file location
df = pd.read_csv(file_path)

# Define the number of hash buckets (reduced dimensions)
n_buckets = 500

# Initialize FeatureHasher
hasher = FeatureHasher(n_features=n_buckets, input_type='pair')

# Initialize an empty list to store (order_id, aggregated features) tuples
order_features_list = []

# Process each unique order
for order_id in df['order'].unique():
    order_data = df[df['order'] == order_id]
    
    # Prepare the data for hashing
    hash_data = []
    for _, row in order_data.iterrows():
        features = [
            ('product_id', row['product']),
            ('customer_id', row['customer']),
            ('color_id', row['color']),
            ('size_id', row['size']),
            ('group_id', row['group'])
        ]
        hash_data.append(features)
    
    # Apply hashing
    hashed_features = hasher.transform(hash_data)
    
    # Convert hashed features to binary (presence/absence)
    hashed_features_binary = (hashed_features != 0).astype(int)
    
    # Aggregate features for this order using max to maintain binary values
    order_hashed_features = hashed_features_binary.max(axis=0)
    
    # Append (order_id, aggregated features) as a tuple
    order_features_list.append((order_id, csr_matrix(order_hashed_features)))

# Sort the list of tuples by order_id
order_features_list.sort(key=lambda x: x[0])

# Extract sorted order IDs and their corresponding feature matrices
sorted_order_ids = [item[0] for item in order_features_list]
sorted_order_features = [item[1] for item in order_features_list]

# Stack the list of sparse matrices vertically
order_features_matrix = vstack(sorted_order_features)

with open("etail/ours/hash_500.pickle", 'wb') as f:
    pickle.dump(order_features_matrix.tocsr(), f)