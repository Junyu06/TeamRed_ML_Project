import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df_multiclass = pd.read_csv('../../ds_salaries.csv')

experience_counts = df_multiclass['experience_level'].value_counts(normalize=True) * 100  # Convert to percentage

# Print the results
print("Experience Level Distribution (%):\n", experience_counts)
