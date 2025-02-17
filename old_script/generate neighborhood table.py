import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


# Load your data
node_table = pd.read_csv('dataset/charlotte/node.csv')
edge_table = pd.read_csv('dataset/charlotte/edge.csv')

# Remove records with NaN values
node_table = node_table.dropna()
edge_table = edge_table.dropna()

# Check for NaN values and report
node_nan_count = node_table.isnull().sum().sum()
edge_nan_count = edge_table.isnull().sum().sum()
print(f"Number of NaN values in node_table after removal: {node_nan_count}")
print(f"Number of NaN values in edge_table after removal: {edge_nan_count}")

# Extract coordinates for spatial indexing
coordinates = node_table[['EASTING', 'NORTHING']].values
node_ids = node_table['NodeID'].values
elevations = node_table['Elevation'].values
eastings = node_table['EASTING'].values
northings = node_table['NORTHING'].values

# Build spatial index using cKDTree
tree = cKDTree(coordinates)

# Convert edge table to a set of tuples for fast lookup
edge_set = set(zip(edge_table['From_NodeID'], edge_table['To_NodeID']))

# Use a set to track added edges
added_edges = set()

# Find neighbors within 700 feet (213.36 meters)
neighborhood_data = []
radius = 700
for idx, (node_id, coord, elevation, easting, northing) in enumerate(zip(node_ids, coordinates, elevations, eastings, northings)):
    indices = tree.query_ball_point(coord, r=radius)
    for neighbor_idx in indices:
        if neighbor_idx != idx:  # Exclude self-loops
            neighbor_id = node_ids[neighbor_idx]
            neighbor_elevation = elevations[neighbor_idx]
            neighbor_easting = eastings[neighbor_idx]
            neighbor_northing = northings[neighbor_idx]
            distance = np.linalg.norm(coord - coordinates[neighbor_idx])
            elevation_diff = abs(elevation - neighbor_elevation)
            is_connected = 1 if (node_id, neighbor_id) in edge_set or (neighbor_id, node_id) in edge_set else 0

            # Use a sorted tuple to ensure uniqueness (a, b) == (b, a)
            edge = tuple(sorted((node_id, neighbor_id)))
            if edge not in added_edges:
                neighborhood_data.append([
                    node_id, neighbor_id, round(distance, 3), round(elevation_diff, 3), is_connected,
                    round(easting, 3), round(northing, 3), round(neighbor_easting, 3), round(neighbor_northing, 3)
                ])
                added_edges.add(edge)

# Convert to NumPy array
neighborhood_array = np.array(neighborhood_data)

# Save as a .npy file
np.save('neighborhood_table.npy', neighborhood_array)

# Save as a CSV file
neighborhood_df = pd.DataFrame(neighborhood_data, columns=['From_NodeID', 'To_NodeID', 'Distance', 'Elevation_Diff', 'Is_Connected', 'From_Easting', 'From_Northing', 'To_Easting', 'To_Northing'])
neighborhood_df = neighborhood_df.astype({'From_NodeID': 'int', 'To_NodeID': 'int', 'Is_Connected': 'int'})
neighborhood_df.to_csv('neighborhood_table.csv', index=False)

# Save positive neighborhood table as a CSV file
positive_neighborhood_df = neighborhood_df[neighborhood_df['Is_Connected'] == 1]
positive_neighborhood_df.to_csv('positive_neighborhood_table.csv', index=False)

# Generate histogram data
bins = np.arange(0, 750, 50)
neighborhood_df['Bin'] = pd.cut(neighborhood_df['Distance'], bins=bins, right=False, labels=bins[:-1])
histogram_df = neighborhood_df.groupby('Bin').agg(
    Positive=('Is_Connected', 'sum'),
    Negative=('Is_Connected', lambda x: (x == 0).sum()),
    Total=('Is_Connected', 'count')
).reset_index()

# Save histogram data to CSV
histogram_df.to_csv('neighborhood_histogram.csv', index=False)

print("Neighborhood table and histogram saved. Histogram data is in 'neighborhood_histogram.csv'.")
print("Positive neighborhood table saved as 'positive_neighborhood_table.csv'.")

print(len(neighborhood_df))
print(len(positive_neighborhood_df))