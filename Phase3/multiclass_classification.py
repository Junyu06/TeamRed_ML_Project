import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, classification_report
import joblib

# Load data
def load_data(folder='multiclass'):
    X_train = pd.read_csv(f'{folder}/multiclass_X_train_scaled.csv')
    y_train = pd.read_csv(f'{folder}/multiclass_y_train.csv').squeeze()
    return X_train, y_train

# Evaluation function (on training set only)
def evaluate_model(name, model, X_train, y_train):
    y_pred = model.predict(X_train)
    f1 = f1_score(y_train, y_pred, average='macro')
    acc = accuracy_score(y_train, y_pred)
    print(f"{name} - Training Macro F1 Score: {f1:.4f}, Accuracy: {acc:.4f}")
    print(classification_report(y_train, y_pred))
    return f1, acc

def main():
    X_train, y_train = load_data()

    models = {
        "Decision Tree": (DecisionTreeClassifier(), {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10]
        }),

        "KNN": (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }),

        "Random Forest": (RandomForestClassifier(random_state=42), {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20]
        }),

        "XGBoost": (XGBClassifier(objective='multi:softprob', num_class=4, eval_metric='mlogloss'), {
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 6],
            'reg_alpha': [0, 0.5],
            'reg_lambda': [1, 2]
        })
    }

    results = []
    for name, (model, params) in models.items():
        print(f"\nTuning and training {name}...")
        grid = GridSearchCV(model, params, cv=3, scoring='f1_macro', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        joblib.dump(best_model, f'multiclass/{name.replace(" ", "_").lower()}_model.pkl')
        f1, acc = evaluate_model(name, best_model, X_train, y_train)
        results.append((name, f1, acc))

    print("\nTraining Summary:")
    for name, f1, acc in results:
        print(f"{name}: F1={f1:.4f}, Accuracy={acc:.4f}")

if __name__ == '__main__':
    main()
