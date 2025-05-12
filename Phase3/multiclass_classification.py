import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, classification_report


def load_data(folder='multiclass'):
    X_train = pd.read_csv(f'{folder}/multiclass_X_train_scaled.csv')
    y_train = pd.read_csv(f'{folder}/multiclass_y_train.csv').squeeze()
    return X_train, y_train


def evaluate_model(name, model, X_train, y_train):
    y_pred = model.predict(X_train)
    f1 = f1_score(y_train, y_pred, average='macro')
    acc = accuracy_score(y_train, y_pred)
    print(f"{name} - Training Macro F1 Score: {f1:.4f}, Accuracy: {acc:.4f}")
    print(classification_report(y_train, y_pred))
    return f1, acc


def plot_cv_results(cv_results_df, model_name, score_name='mean_test_score'):
    filename = f"multiclass/{model_name}_cv_plot.png"
    try:
        if 'param_max_depth' in cv_results_df.columns and 'param_n_estimators' in cv_results_df.columns:
            pivot = cv_results_df.pivot(index='param_max_depth', columns='param_n_estimators', values=score_name)
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title(f"{model_name.title()} - F1 Macro Heatmap")
            plt.xlabel("Number of Estimators")
            plt.ylabel("Max Depth")

        elif 'param_learning_rate' in cv_results_df.columns and 'param_max_depth' in cv_results_df.columns:
            pivot = cv_results_df.pivot(index='param_max_depth', columns='param_learning_rate', values=score_name)
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title(f"{model_name.title()} - F1 Macro Heatmap")
            plt.xlabel("Learning Rate")
            plt.ylabel("Max Depth")

        elif 'param_n_neighbors' in cv_results_df.columns and 'param_weights' in cv_results_df.columns:
            pivot = cv_results_df.pivot(index='param_n_neighbors', columns='param_weights', values=score_name)
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title(f"{model_name.title()} - F1 Macro Heatmap")
            plt.xlabel("Weights")
            plt.ylabel("Number of Neighbors")

        elif 'param_max_depth' in cv_results_df.columns and 'param_min_samples_split' in cv_results_df.columns:
            pivot = cv_results_df.pivot(index='param_max_depth', columns='param_min_samples_split', values=score_name)
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title(f"{model_name.title()} - F1 Macro Heatmap")
            plt.xlabel("Min Samples Split")
            plt.ylabel("Max Depth")

        else:
            print(f"No compatible parameters found for plotting {model_name}")
            return

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Saved CV plot to {filename}")

    except Exception as e:
        print(f"Failed to generate plot for {model_name}: {e}")


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

        "XGBoost": (XGBClassifier(objective='multi:softprob',
                                  num_class=4,
                                  eval_metric='mlogloss'), {
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 6],
            'reg_alpha': [0, 0.5],
            'reg_lambda': [1, 2]
        })
    }

    results = []
    for name, (model, params) in models.items():
        print(f"\nTuning and training {name}...")
        start = time.time()
        grid = GridSearchCV(model, params, cv=3, scoring='f1_macro', n_jobs=-1)
        grid.fit(X_train, y_train)
        end = time.time()
        tuning_time = end - start

        best_model = grid.best_estimator_
        model_filename = f'multiclass/{name.replace(" ", "_").lower()}_model.pkl'
        joblib.dump(best_model, model_filename)
        print(f"Saved best model to {model_filename}")

        cv_results_df = pd.DataFrame(grid.cv_results_)
        cv_results_path = f'multiclass/{name.replace(" ", "_").lower()}_cv_results.csv'
        cv_results_df.to_csv(cv_results_path, index=False)
        print(f"Saved CV results to {cv_results_path}")

        plot_cv_results(cv_results_df, name.replace(" ", "_").lower())

        f1, acc = evaluate_model(name, best_model, X_train, y_train)
        results.append((name, f1, acc, tuning_time))

    print("\nTraining Summary:")
    for name, f1, acc, tuning_time in results:
        print(f"{name}: F1={f1:.4f}, Accuracy={acc:.4f}, Tuning Time={tuning_time:.2f}s")


if __name__ == '__main__':
    main()
