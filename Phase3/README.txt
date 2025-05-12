TeamRed ML Project - Phase 3
===========================

This project implements both binary and multiclass classification models using various machine learning algorithms.

Project Structure
----------------
- binary/                  # Directory for binary classification results and plots
- multiclass/             # Directory for multiclass classification results and plots
- binary_split.py         # Data preprocessing for binary classification
- binary_classification.py # Binary classification model training
- binary_test.py          # Binary classification model evaluation
- multiclass_split.py     # Data preprocessing for multiclass classification
- multiclass_classification.py # Multiclass classification model training
- multiclass_test.py      # Multiclass classification model evaluation
- main.py                 # Main script to run all analyses
- ml_project.yml          # Conda environment configuration
- breast-cancer-wisconsin-data.csv  # Binary classification dataset
- ds_salaries.csv         # Multiclass classification dataset

Setup Instructions
-----------------
1. Create and activate the conda environment:
   # Create environment from yml file
   conda env create -f ml_project.yml
   
   # Activate the environment
   conda activate ml_project

Running the Analysis
-------------------
1. Ensure the conda environment is activated
2. Run the main script:
   python main.py

Output
------
The script will generate the following outputs:

Binary Classification (in 'binary/' directory):
Data Files:
- binary_X_train.csv, binary_X_test.csv         # Unscaled train/test features
- binary_y_train.csv, binary_y_test.csv         # Unscaled train/test labels
- binary_X_train_scaled.csv, binary_X_test_scaled.csv  # Scaled train/test features

Model Files (for each model: Logistic Regression, SVM, Random Forest, XGBoost):
- *_best_model.pkl                              # Trained model files
- *_cv_results.csv                              # Cross-validation results
- *_cv_plot.png                                 # Hyperparameter tuning plots

Evaluation Plots (in 'binary/plots/' directory):
- *_confusion_matrix.png                        # Confusion matrix plots
- *_roc_curve.png                              # ROC curve plots

Multiclass Classification (in 'multiclass/' directory):
Data Files:
- multiclass_X_train.csv, multiclass_X_test.csv         # Unscaled train/test features
- multiclass_y_train.csv, multiclass_y_test.csv         # Unscaled train/test labels
- multiclass_X_train_scaled.csv, multiclass_X_test_scaled.csv  # Scaled train/test features

Model Files (for each model: Decision Tree, KNN, Random Forest, XGBoost):
- *_model.pkl                                   # Trained model files
- *_cv_results.csv                              # Cross-validation results
- *_cv_plot.png                                 # Hyperparameter tuning plots
- *_confusion_matrix.png                        # Confusion matrix plots

Terminal Output:
- Training progress and model performance metrics
- Classification reports for each model
- Cross-validation scores and best parameters
- Final model evaluation results including:
  * Accuracy scores
  * F1 scores (macro average for multiclass)
  * ROC AUC scores (for binary classification)
  * Detailed classification reports
  * Confusion matrix statistics

Notes
-----
- The binary classification uses the Breast Cancer Wisconsin dataset
  * Target: Diagnosis (Malignant/Benign)
  * Models: Logistic Regression, SVM, Random Forest, XGBoost
  * Evaluation: ROC curves, confusion matrices, classification reports

- The multiclass classification uses the Data Science Salaries dataset
  * Target: Experience Level (EN/MI/SE/EX)
  * Models: Decision Tree, KNN, Random Forest, XGBoost
  * Evaluation: Confusion matrices, classification reports

- All models are trained with:
  * Cross-validation (5-fold for binary, 3-fold for multiclass)
  * Hyperparameter tuning using GridSearchCV
  * Performance metrics saved in CSV files
  * Visualization plots saved as PNG files

- Results are automatically saved in their respective directories
- Progress messages will be displayed in the terminal during execution
- Total execution time will be displayed at the end