import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df_binary = pd.read_csv('../breast-cancer-wisconsin-data.csv')

# Drop unnecessary columns
df_binary = df_binary.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Encode 'diagnosis': Malignant (M) → 1, Benign (B) → 0
df_binary['diagnosis'] = df_binary['diagnosis'].map({'M': 1, 'B': 0})

# Define features (X) and target (y)
X = df_binary.drop(columns=['diagnosis']) 
y = df_binary['diagnosis']

# Split the dataset with stratification (ensuring class ratio consistency)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=89, stratify=y)

# Verify the class distribution in both sets
print("Train set class distribution:\n", y_train.value_counts(normalize=True))
print("Test set class distribution:\n", y_test.value_counts(normalize=True))

# Save the unscaled training and testing sets
X_train.to_csv('binary_X_train.csv', index=False)
y_train.to_csv('binary_y_train.csv', index=False)
X_test.to_csv('binary_X_test.csv', index=False)
y_test.to_csv('binary_y_test.csv', index=False)

print("Unscaled datasets saved successfully!")

# --- APPLY FEATURE SCALING AFTER SPLITTING ---
# Initialize the scaler
scaler = StandardScaler()

# Fit on training data and transform both training and testing sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use the same transformation for test set

# Convert scaled arrays back to DataFrames
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Save the scaled training and testing sets
X_train_scaled.to_csv('binary_X_train_scaled.csv', index=False)
X_test_scaled.to_csv('binary_X_test_scaled.csv', index=False)

print("Feature scaling applied and scaled datasets saved successfully!")
