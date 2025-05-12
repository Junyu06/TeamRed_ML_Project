import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, RocCurveDisplay,
    classification_report
)

MODELS = [
    'logistic_regression',
    'svm',
    'random_forest',
    'xgboost'
]

def load_test_data(folder='binary'):
    X_test = pd.read_csv(f'{folder}/binary_X_test_scaled.csv')
    y_test = pd.read_csv(f'{folder}/binary_y_test.csv').squeeze()
    return X_test, y_test

def evaluate_and_plot(model_name, model, X_test, y_test, output_dir='binary/plots'):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Classification report
    print(f"\nClassification Report for {model_name.capitalize().replace('_', ' ')}:\n")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm_disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    cm_disp.ax_.set_title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.title(f'{model_name} - ROC Curve (AUC={roc_auc:.2f})')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_roc_curve.png")
    plt.close()

    print(f"Saved plots for {model_name}.\n")

def main():
    os.makedirs('binary/plots', exist_ok=True)
    X_test, y_test = load_test_data()

    for model_name in MODELS:
        print(f"\nTesting model: {model_name}")
        model = joblib.load(f'binary/{model_name}_best_model.pkl')
        evaluate_and_plot(model_name, model, X_test, y_test)

if __name__ == "__main__":
    main()
