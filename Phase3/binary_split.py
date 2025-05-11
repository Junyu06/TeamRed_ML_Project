import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_and_scale_binary_data(input_path='breast-cancer-wisconsin-data.csv', output_dir='binary', random_state=89):
    """
    Splits and scales the Breast Cancer dataset for binary classification.
    Saves both unscaled and scaled train/test sets to CSV files in the specified output directory.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(input_path)

    # Drop unnecessary columns
    df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

    # Encode diagnosis: M → 1 (Malignant), B → 0 (Benign)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Define features and target
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Save unscaled datasets
    X_train.to_csv(f'{output_dir}/binary_X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/binary_X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/binary_y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/binary_y_test.csv', index=False)
    print("Unscaled datasets saved in 'binary/'")

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Save scaled datasets
    X_train_scaled.to_csv(f'{output_dir}/binary_X_train_scaled.csv', index=False)
    X_test_scaled.to_csv(f'{output_dir}/binary_X_test_scaled.csv', index=False)
    print("Scaled datasets saved in 'binary/'")

if __name__ == "__main__":
    split_and_scale_binary_data()
