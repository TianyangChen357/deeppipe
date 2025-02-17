import geopandas as gpd
import rasterio
from shapely.geometry import Point
import pandas as pd
import os
import time

# Define file paths
data_folder = "0_data"
table_folder = "1_table"
dem_file = os.path.join(data_folder, "mecklenburg-DEM03.tif")
storm_structures_file = os.path.join(data_folder, "Storm_Structures_subset.shp")
roads_file = os.path.join(data_folder, "Streets.shp")
streams_file = os.path.join(data_folder, "Streams.shp")
output_csv = os.path.join("storm_structures_analysis.csv")

# Start timing
start_time = time.time()

# Load storm structures
load_start = time.time()
storm_structures = gpd.read_file(storm_structures_file)
print(f"Loaded storm structures in {time.time() - load_start:.3f} seconds")

# Load road network and stream network
load_start = time.time()
roads = gpd.read_file(roads_file)
streams = gpd.read_file(streams_file)
print(f"Loaded roads and streams in {time.time() - load_start:.3f} seconds")

# Load DEM
load_start = time.time()
with rasterio.open(dem_file) as dem:
    # Get elevation for each storm structure
    storm_structures["elevation"] = [round(float(list(dem.sample([(geom.x, geom.y)]))[0][0]), 3) for geom in storm_structures.geometry]
print(f"Extracted elevation in {time.time() - load_start:.3f} seconds")

# Function to calculate the nearest distance
def nearest_distance(geom, gdf):
    return round(float(gdf.geometry.distance(geom).min()), 3)

# Calculate distances to streams and roads
distance_start = time.time()
storm_structures["distance_to_stream"] = storm_structures.geometry.apply(lambda geom: nearest_distance(geom, streams))
storm_structures["distance_to_road"] = storm_structures.geometry.apply(lambda geom: nearest_distance(geom, roads))
print(f"Calculated distances in {time.time() - distance_start:.3f} seconds")

# Extract x, y coordinates
coord_start = time.time()
storm_structures["x"] = storm_structures.geometry.x.round(3).astype(float)
storm_structures["y"] = storm_structures.geometry.y.round(3).astype(float)
print(f"Extracted coordinates in {time.time() - coord_start:.3f} seconds")

# Select required columns
select_start = time.time()
storm_structures["ASSETID"] = storm_structures["ITPIPE_ASS"]
output_df = storm_structures[["x", "y", "ASSETID", "elevation", "distance_to_stream", "distance_to_road"]]
print(f"Selected required columns in {time.time() - select_start:.3f} seconds")

# Save to CSV
save_start = time.time()
output_df.to_csv(output_csv, index=False, float_format='%.3f')
print(f"Saved CSV in {time.time() - save_start:.3f} seconds")

# Total execution time
print(f"Total execution time: {time.time() - start_time:.3f} seconds")
print(f"Table saved as {output_csv}")
