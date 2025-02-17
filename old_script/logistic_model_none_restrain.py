import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Load your data
edge_table = pd.read_csv("dataset/charlotte/edge.csv")
node_table = pd.read_csv('dataset/charlotte/node.csv')

# Remove records with NaN values
edge_table = edge_table.dropna()
node_table = node_table.dropna()

# Check for NaN values and report
edge_nan_count = edge_table.isnull().sum().sum()
node_nan_count = node_table.isnull().sum().sum()
print(f"Number of NaN values in edge_table after removal: {edge_nan_count}")
print(f"Number of NaN values in node_table after removal: {node_nan_count}")

# Sampling positive samples
positive_samples = edge_table.sample(n=5000, replace=True, random_state=42)

# Merge to compute features for positive samples
positive_samples = positive_samples.merge(node_table, left_on='From_NodeID', right_on='NodeID')
positive_samples = positive_samples.merge(node_table, left_on='To_NodeID', right_on='NodeID', suffixes=('_from', '_to'))

# Calculate distance and elevation difference
positive_samples['distance'] = np.sqrt((positive_samples['EASTING_from'] - positive_samples['EASTING_to'])**2 +
                                       (positive_samples['NORTHING_from'] - positive_samples['NORTHING_to'])**2)
positive_samples['elevation_diff'] = abs(positive_samples['Elevation_from'] - positive_samples['Elevation_to'])

# Label as positive
positive_samples['label'] = 1

# Generate negative samples based on nodes within 700 feet
negative_samples = []
for _, from_node in node_table.iterrows():
    nearby_nodes = node_table[
        (np.sqrt((node_table['EASTING'] - from_node['EASTING'])**2 +
                 (node_table['NORTHING'] - from_node['NORTHING'])**2) <= 700) &
        (node_table['NodeID'] != from_node['NodeID'])
    ]
    for _, to_node in nearby_nodes.iterrows():
        distance = np.sqrt((from_node['EASTING'] - to_node['EASTING'])**2 +
                           (from_node['NORTHING'] - to_node['NORTHING'])**2)
        elevation_diff = abs(from_node['Elevation'] - to_node['Elevation'])
        negative_samples.append([from_node['NodeID'], to_node['NodeID'], distance, elevation_diff])

negative_samples = pd.DataFrame(negative_samples, columns=['From_NodeID', 'To_NodeID', 'distance', 'elevation_diff'])
negative_samples['label'] = 0

# Combine positive and negative samples
data = pd.concat([
    positive_samples[['distance', 'elevation_diff', 'label']],
    negative_samples[['distance', 'elevation_diff', 'label']]
])

# Split data for training and validation
positive_train = positive_samples[~positive_samples.index.isin(positive_samples.sample(n=1000, random_state=42).index)]
negative_train = negative_samples[~negative_samples.index.isin(negative_samples.sample(n=1000, random_state=42).index)]

training_data = pd.concat([positive_train, negative_train])
validation_data = pd.concat([positive_samples.sample(n=1000, random_state=42), negative_samples.sample(n=1000, random_state=42)])

X_train = training_data[['distance', 'elevation_diff']]
y_train = training_data['label']
X_test = validation_data[['distance', 'elevation_diff']]
y_test = validation_data['label']

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Print model coefficients
print("Model Coefficients:")
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Output confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Add predictions to validation dataset
validation_data['y_pred'] = y_pred

# Export validation dataset to CSV
validation_data.to_csv('validation_data_700ft.csv', index=False)

