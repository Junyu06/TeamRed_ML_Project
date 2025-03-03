import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

#import dataset
X = pd.read_csv('binary_X_train_scaled.csv')
y = pd.read_csv('binary_y_train.csv')
#df_multi = pd.read_csv('../ds_salaries.csv')

def check_data(X, y):
    print(f'Head:\n{X.head()}')
    print(f'\nDataset Info:\n{X.info()}')
    print(f'\n\n\nMissing values:\n{X.isnull().sum()}')
    print(f'\n\n\nClass distribution:\n{y.value_counts(normalize=True)}')

check_data(X,y)

def EDA(X, y):
    # Combine features and labels for analysis
    df_train = X.copy()
    df_train['diagnosis'] = y

    # Summary statistics
    print(df_train.describe())

    # Diagnosis distribution
    sns.countplot(data=df_train, x='diagnosis')
    plt.title('Diagnosis Distribution')
    plt.show()

    # Compute correlation matrix
    corr_matrix = df_train.corr()

    # Plot heatmap
    plt.figure(figsize=(12,10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.show()

    # Pairwise relationships for key features
    key_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']
    key_features = [f for f in key_features if f in df_train.columns]  # Ensure features exist

    if key_features:
        sns.pairplot(df_train, vars=key_features, hue='diagnosis')
        plt.show()
    else:
        print("Warning: Some key features are missing from the dataset!")

# Perform EDA on the loaded training set
EDA(X,y)

