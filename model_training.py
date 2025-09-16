# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

print("Starting model training process...")

# Loading Data
# Loading the dataset from the CSV file.
try:
    df = pd.read_csv('fruit_data.csv')
    # Dropping the first unnamed column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
except FileNotFoundError:
    print("Error: 'fruit_data.csv' not found. Make sure the file is in the correct directory.")
    exit()

print("Data loaded successfully.")
print("Data head:\n", df.head())

# Data Cleaning         
# I noticed 'Largee' in your data snippet, which looks like a typo.
df['size'] = df['size'].replace('Largee', 'Large')
print("\nCleaned unique sizes:", df['size'].unique())


# Feature Engineering & Preprocessing
# Define our features (X) and the target (y)
features = ['color', 'size', 'weight']
target = 'fruit_type'

X = df[features]
y = df[target]

# Define the transformations for the columns.
# - 'color' is nominal (no order), so I use OneHotEncoder.
# - 'size' is ordinal (has a clear order), so I use OrdinalEncoder.
# - 'weight' is numeric and doesn't need a transformer here, so I use 'passthrough'.

# Define the specific order for the 'size' category
size_categories = ['Tiny', 'Small', 'Medium', 'Large']

preprocessor = ColumnTransformer(
    transformers=[
        ('color_ohe', OneHotEncoder(handle_unknown='ignore'), ['color']),
        ('size_ordinal', OrdinalEncoder(categories=[size_categories]), ['size'])
    ],
    remainder='passthrough' # Keep the 'weight' column as is
)

# Model Training
# I used a RandomForestClassifier, which is a great all-around model for this kind of task.
# I created a Pipeline to chain the preprocessing and the model together.
# This makes it easy to manage and prevents data leakage.
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining the model...")
# Training the entire pipeline on training data
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Evaluation
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.4f}")


# Save the Model
# Save the trained pipeline to a file so the web app can use it.
# joblib is generally preferred for saving sklearn models.
joblib.dump(model_pipeline, 'fruit_pipeline.joblib')
print("\nModel pipeline saved to 'fruit_pipeline.joblib'")
print("Script finished successfully.")