import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, roc_auc_score, classification_report

def load_binary_data(folder='binary'):
    X_train = pd.read_csv(f'{folder}/binary_X_train_scaled.csv')
    y_train = pd.read_csv(f'{folder}/binary_y_train.csv').squeeze()
    X_test = pd.read_csv(f'{folder}/binary_X_test_scaled.csv')
    y_test = pd.read_csv(f'{folder}/binary_y_test.csv').squeeze()
    return X_train, X_test, y_train, y_test

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"\n{name} Performance:")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return recall, roc_auc

def run_binary_models():
    X_train, X_test, y_train, y_test = load_binary_data()

    models = {
        'Logistic Regression': GridSearchCV(
            Pipeline([('clf', LogisticRegression(solver='liblinear'))]),
            param_grid={
                'clf__C': [0.01, 0.1, 1, 10],
                'clf__penalty': ['l1', 'l2']
            },
            cv=5,
            scoring='recall'
        ),

        'SVM': GridSearchCV(
            Pipeline([('clf', SVC(probability=True))]),
            param_grid={
                'clf__C': [0.1, 1, 10],
                'clf__kernel': ['linear', 'rbf'],
                'clf__gamma': ['scale', 'auto']
            },
            cv=5,
            scoring='recall'
        ),

        'Random Forest': GridSearchCV(
            RandomForestClassifier(),
            param_grid={
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10]
            },
            cv=5,
            scoring='recall'
        ),

        'XGBoost': GridSearchCV(
            XGBClassifier(eval_metric='logloss'),
            param_grid={
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [1, 10]
            },
            cv=5,
            scoring='recall'
        )
    }

    results = []
    for name, model in models.items():
        print(f"\nTraining {name} with hyperparameter tuning...")
        model.fit(X_train, y_train)

        # Print best hyperparameters and best cross-validated recall
        print(f"Best hyperparameters for {name}: {model.best_params_}")
        print(f"Best cross-validated recall: {model.best_score_:.4f}")

        # Save best model to file
        model_filename = f"binary/{name.lower().replace(' ', '_')}_best_model.pkl"
        joblib.dump(model.best_estimator_, model_filename)
        print(f"Saved best model to {model_filename}")

        # Evaluate on test set
        recall, roc_auc = evaluate_model(name, model, X_test, y_test)
        results.append((name, recall, roc_auc))


    print("\nSummary:")
    for name, recall, auc in results:
        print(f"{name:20} | Recall: {recall:.4f} | ROC AUC: {auc:.4f}")

if __name__ == "__main__":
    run_binary_models()
