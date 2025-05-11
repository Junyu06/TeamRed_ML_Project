import pandas as pd
import joblib
import os
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load test data
def load_test_data(folder='multiclass'):
    X_test = pd.read_csv(f'{folder}/multiclass_X_test_scaled.csv')
    y_test = pd.read_csv(f'{folder}/multiclass_y_test.csv').squeeze()
    return X_test, y_test

# Evaluation with saved confusion matrix
def evaluate_model(name, model, X_test, y_test, output_dir='multiclass'):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} - Test Macro F1 Score: {f1:.4f}, Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save confusion matrix plot silently
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{name.replace(' ', '_').lower()}_confusion_matrix.png")
    plt.close()

    return f1, acc

def main():
    X_test, y_test = load_test_data()

    model_files = [
        ("Decision Tree", "multiclass/decision_tree_model.pkl"),
        ("KNN", "multiclass/knn_model.pkl"),
        ("Random Forest", "multiclass/random_forest_model.pkl"),
        ("XGBoost", "multiclass/xgboost_model.pkl")
    ]

    results = []
    for name, file in model_files:
        model = joblib.load(file)
        f1, acc = evaluate_model(name, model, X_test, y_test)
        results.append((name, f1, acc))

    print("\nTest Summary:")
    for name, f1, acc in results:
        print(f"{name}: F1={f1:.4f}, Accuracy={acc:.4f}")

if __name__ == '__main__':
    main()
