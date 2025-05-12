import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_cv_results(cv_results_df, model_name, score_name='mean_test_score'):
    """Generates tuning plot depending on available parameters."""
    filename = f"binary/{model_name}_cv_plot.png"

    try:
        if 'param_clf__C' in cv_results_df.columns:
            # Logistic Regression or SVM
            if 'param_clf__penalty' in cv_results_df.columns:
                sns.lineplot(
                    x='param_clf__C',
                    y=score_name,
                    hue='param_clf__penalty',
                    data=cv_results_df,
                    marker='o'
                )
            else:
                sns.lineplot(
                    x='param_clf__C',
                    y=score_name,
                    data=cv_results_df,
                    marker='o'
                )
            plt.title(f"{model_name.title()} - Cross-Validated Recall")
            plt.xlabel("C")
            plt.ylabel("Recall")
            plt.savefig(filename)
            plt.close()

        elif 'param_max_depth' in cv_results_df.columns and 'param_n_estimators' in cv_results_df.columns:
            # Random Forest
            pivot = cv_results_df.pivot(
                index='param_max_depth',
                columns='param_n_estimators',
                values=score_name
            )
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title(f"{model_name.title()} - Recall Heatmap")
            plt.xlabel("Number of Estimators")
            plt.ylabel("Max Depth")
            plt.savefig(filename)
            plt.close()

        elif 'param_learning_rate' in cv_results_df.columns and 'param_max_depth' in cv_results_df.columns:
            # XGBoost
            pivot = cv_results_df.pivot_table(
                index='param_max_depth',
                columns='param_learning_rate',
                values=score_name,
                aggfunc='mean'
            )
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title(f"{model_name.title()} - Recall Heatmap")
            plt.xlabel("Learning Rate")
            plt.ylabel("Max Depth")
            plt.savefig(filename)
            plt.close()

    except Exception as e:
        print(f"Could not generate plot for {model_name}: {e}")
    else:
        print(f"Saved tuning plot to {filename}")

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
        print(f"Hyperparameter grid for {name}: {model.param_grid}")

        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        print(f"Tuning time: {end - start:.2f} seconds")

        print(f"Best hyperparameters for {name}: {model.best_params_}")
        print(f"Best cross-validated recall: {model.best_score_:.4f}")

        # Save model
        model_filename = f"binary/{name.lower().replace(' ', '_')}_best_model.pkl"
        joblib.dump(model.best_estimator_, model_filename)
        print(f"Saved best model to {model_filename}")

        # Save CV results
        cv_results_df = pd.DataFrame(model.cv_results_)
        cv_filename = f"binary/{name.lower().replace(' ', '_')}_cv_results.csv"
        cv_results_df.to_csv(cv_filename, index=False)
        print(f"Saved CV results to {cv_filename}")

        # Plot tuning curves (basic support for LR and RF-style grids)
        plot_cv_results(cv_results_df, name.lower().replace(' ', '_'))

        # Evaluate test performance
        recall, roc_auc = evaluate_model(name, model, X_test, y_test)
        results.append((name, recall, roc_auc))

    print("\nSummary:")
    for name, recall, auc in results:
        print(f"{name:20} | Recall: {recall:.4f} | ROC AUC: {auc:.4f}")


if __name__ == "__main__":
    run_binary_models()
