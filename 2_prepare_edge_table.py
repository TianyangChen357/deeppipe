import pandas as pd
import os
import numpy as np
# Define file paths
data_folder = "0_data"
storm_pipes_file = os.path.join(data_folder, "Storm_Pipes.csv")
storm_structures_file = os.path.join("storm_structures_analysis.csv")
output_pipes_file = os.path.join("storm_pipes_analysis.csv")
output_pipes_table_file=os.path.join("edge_table.csv")
# Load storm pipes
storm_pipes = pd.read_csv(storm_pipes_file, dtype=str)

# Load storm structures
storm_structures = pd.read_csv(storm_structures_file, dtype=str)
storm_structures[["x", "y", "elevation"]] = storm_structures[["x", "y", "elevation"]].astype(float)
# Get valid structure IDs
valid_structure_ids = set(storm_structures["ASSETID"].unique())

# Filter pipes where both upstream and downstream structures exist in the structure table
storm_pipes_filtered = storm_pipes[
    (storm_pipes["US_ASSETID"].isin(valid_structure_ids)) & (storm_pipes["DS_ASSETID"].isin(valid_structure_ids))
].copy()

# Merge upstream structure attributes
storm_pipes_filtered = storm_pipes_filtered.merge(
    storm_structures, left_on="US_ASSETID", right_on="ASSETID", suffixes=("", "_us"), how="left"
)
# print(storm_pipes_filtered.keys())
# Merge downstream structure attributes
storm_pipes_filtered = storm_pipes_filtered.merge(
    storm_structures, left_on="DS_ASSETID", right_on="ASSETID", suffixes=("", "_ds"), how="left"
)

# print(storm_pipes_filtered.keys())
# Select required columns: ITPIPE_ASSETID, US_ASSETID, DS_ASSETID, and all _us and _ds attributes
columns_to_keep = ["ITPIPE_ASSETID", "US_ASSETID", "DS_ASSETID","x","y","elevation","distance_to_stream","distance_to_road"] + [col for col in storm_pipes_filtered.columns if col.endswith("_us") or col.endswith("_ds")]
storm_pipes_filtered = storm_pipes_filtered[columns_to_keep]
storm_pipes_filtered.drop(columns=["ASSETID_ds"])

storm_pipes_filtered["elevation_dif"] = abs(storm_pipes_filtered["elevation"]-storm_pipes_filtered["elevation_ds"]).round(3)
# Compute Euclidean distance for pipe length
storm_pipes_filtered["pipe_length"] = np.sqrt(
    (storm_pipes_filtered["x"] - storm_pipes_filtered["x_ds"])**2 +
    (storm_pipes_filtered["y"] - storm_pipes_filtered["y_ds"])**2
).round(3)

# Save the cleaned storm pipes table
storm_pipes_filtered.to_csv(output_pipes_file, index=False)
print(storm_pipes_filtered.keys())
edge_table = storm_pipes_filtered[["ITPIPE_ASSETID", "US_ASSETID", "DS_ASSETID", "elevation_dif", "pipe_length"]]

edge_table.to_csv(output_pipes_table_file,index=False)

print(f"Filtered storm pipes table saved as {output_pipes_file}")
