import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_and_scale_multiclass_data(input_path='ds_salaries.csv', output_dir='multiclass', random_state=42):
    """
    Splits and scales the Data Science Job Salaries dataset for multiclass classification.
    Target: experience_level
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(input_path)

    # Drop unnecessary index if exists
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    # Encode ordinal and categorical features
    df['experience_level'] = df['experience_level'].map({'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3})
    df['employment_type'] = df['employment_type'].astype('category').cat.codes

    categorical_cols = ['company_location', 'company_size', 'job_title', 'employee_residence', 'salary_currency']
    df = pd.get_dummies(df, columns=categorical_cols)

    # Define features and target
    X = df.drop(columns=['experience_level', 'salary', 'salary_in_usd'])
    y = df['experience_level']

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.21, random_state=random_state, stratify=y
    )

    # Save unscaled datasets
    X_train.to_csv(f'{output_dir}/multiclass_X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/multiclass_X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/multiclass_y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/multiclass_y_test.csv', index=False)
    print("Unscaled datasets saved in 'multiclass/'")

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Save scaled datasets
    X_train_scaled.to_csv(f'{output_dir}/multiclass_X_train_scaled.csv', index=False)
    X_test_scaled.to_csv(f'{output_dir}/multiclass_X_test_scaled.csv', index=False)
    print("Scaled datasets saved in 'multiclass/'")

if __name__ == "__main__":
    split_and_scale_multiclass_data()
