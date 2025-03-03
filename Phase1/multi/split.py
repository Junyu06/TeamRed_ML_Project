import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df_multiclass = pd.read_csv('../../ds_salaries.csv')

y = df_multiclass['salary_in_usd']
X = df_multiclass.drop(columns=['salary_in_usd'])#keep everything except the target variable

#perform asign to interger for ordinal 
experience_mapping = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
X['experience_level'] = X['experience_level'].map(experience_mapping)

#try to do one hot
categorical_cols = X.select_dtypes(include=['object']).columns
print("Categorical columns: ", categorical_cols)
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

#split
if 'Unnamed: 0' in X.columns:
    X = X.drop(columns=['Unnamed: 0'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=89)
print("Training set size:", X_train.shape, "Test set size:", X_test.shape)

#scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

X_train_scaled.to_csv('multiclass_X_train_scaled.csv', index=False)
X_test_scaled.to_csv('multiclass_X_test_scaled.csv', index=False)
y_train.to_csv('multiclass_y_train.csv', index=False)
y_test.to_csv('multiclass_y_test.csv', index=False)