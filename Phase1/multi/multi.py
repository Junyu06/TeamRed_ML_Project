import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
X = pd.read_csv('multiclass_X_train_scaled.csv')
y = pd.read_csv('multiclass_y_train.csv')

X = X.loc[:, ~X.columns.str.contains('Unnamed')]
y = y.loc[:, ~y.columns.str.contains('Unnamed')]

def check_data(X, y):
    print(f'Head:\n{X.head()}')
    print(f'\nDataset Info:\n{X.info()}')
    print(f'\nMissing values:\n{X.isnull().sum()}')
    print(f'\nTarget variable statistics:\n{y.describe()}')

check_data(X, y)

def EDA(X, y):
    # Combine features and labels for analysis
    df_train = X.copy()
    df_train['salary_in_usd'] = y  # Use salary as the target

    # Summary statistics
    print(df_train.describe())

    # Salary distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df_train['salary_in_usd'], bins=30, kde=True)
    plt.title('Salary Distribution')
    plt.xlabel('Salary in USD')
    plt.ylabel('Frequency')
    plt.show()

    # Boxplot to show salary distribution across job categories
    if 'job_title' in X.columns:
        plt.figure(figsize=(14, 6))
        sns.boxplot(x=df_train['job_title'], y=df_train['salary_in_usd'])
        plt.xticks(rotation=90)
        plt.title('Salary Distribution Across Job Titles')
        plt.show()

    # Compute correlation matrix
    corr_matrix = df_train.corr()

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.show()

    # Pairwise relationships for key features
    numeric_features = df_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Select up to 4 numeric features for pairplot (excluding the target variable)
    key_features = [f for f in numeric_features if f != 'salary_in_usd'][:4]

    if key_features:
        sns.pairplot(df_train, vars=key_features, hue='salary_in_usd')
        plt.show()
    else:
        print("Warning: No suitable numeric features found for pairwise plots!")

# Perform EDA on the loaded training set
EDA(X, y)
