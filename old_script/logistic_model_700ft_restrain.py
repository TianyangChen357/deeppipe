import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the CSV file
file_path = "neighborhood_table.csv"
df = pd.read_csv(file_path)

# Ensure required columns exist
required_columns = ['Is_Connected', 'Elevation_Diff', 'Distance']
if not all(column in df.columns for column in required_columns):
    raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

# Separate the data into positive and negative samples
positive_samples = df[df['Is_Connected'] == 1]
negative_samples = df[df['Is_Connected'] == 0]

# Randomly sample 5000 from each group for training
positive_samples_sampled_train = positive_samples.sample(n=5000, random_state=42)
negative_samples_sampled_train = negative_samples.sample(n=5000, random_state=42)

# Combine the sampled data for training
train_data = pd.concat([positive_samples_sampled_train, negative_samples_sampled_train])

# Prepare the independent variables (X) and dependent variable (y) for training
X_train = train_data[['Elevation_Diff', 'Distance']]
y_train = train_data['Is_Connected']

# Generate random testing dataset with 5000 positive and 5000 negative samples
np.random.seed(42)
test_positive = pd.DataFrame({
    'Is_Connected': [1] * 5000,
    'Elevation_Diff': np.random.uniform(df['Elevation_Diff'].min(), df['Elevation_Diff'].max(), 5000),
    'Distance': np.random.uniform(df['Distance'].min(), df['Distance'].max(), 5000)
})
test_negative = pd.DataFrame({
    'Is_Connected': [0] * 5000,
    'Elevation_Diff': np.random.uniform(df['Elevation_Diff'].min(), df['Elevation_Diff'].max(), 5000),
    'Distance': np.random.uniform(df['Distance'].min(), df['Distance'].max(), 5000)
})
test_data = pd.concat([test_positive, test_negative])

# Prepare the independent variables (X) and dependent variable (y) for testing
X_test = test_data[['Elevation_Diff', 'Distance']]
y_test = test_data['Is_Connected']

# Initialize the logistic regression model
log_reg = LogisticRegression(random_state=42, max_iter=1000)

# Perform 5-fold cross-validation on the training set
cv_scores = cross_val_score(log_reg, X_train, y_train, cv=5, scoring='accuracy')

# Train the logistic regression model
log_reg.fit(X_train, y_train)

# Evaluate on the testing dataset
y_pred_test = log_reg.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
test_report = classification_report(y_test, y_pred_test)

# Display results
print("Cross-Validation Accuracy Scores (Training):", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
print("Standard Deviation (CV):", cv_scores.std())
print("\nTesting Accuracy:", test_accuracy)
print("\nClassification Report (Testing):\n", test_report)

conf_matrix = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix (NumPy Array):")
print(conf_matrix)

coefficients = log_reg.coef_[0]
intercept = log_reg.intercept_[0]
feature_names = ['Elevation_Diff', 'Distance']
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {intercept:.4f}")

