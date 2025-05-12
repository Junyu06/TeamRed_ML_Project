import os
import time
from datetime import datetime
from binary_split import split_and_scale_binary_data
from binary_classification import run_binary_models
from binary_test import main as run_binary_test
from multiclass_split import split_and_scale_multiclass_data
from multiclass_classification import main as run_multiclass_models
from multiclass_test import main as run_multiclass_test

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def print_step(step_num, description):
    """Print a formatted step message."""
    print(f"\nStep {step_num}: {description}")
    print("-" * 60)

def main():
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print_section_header(f"TeamRed ML Project - Phase 3")
    print(f"Started at: {timestamp}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Create necessary directories
    print("\nCreating output directories...")
    os.makedirs('binary', exist_ok=True)
    os.makedirs('multiclass', exist_ok=True)
    os.makedirs('binary/plots', exist_ok=True)
    print("✓ Directories created successfully")
    
    # Binary Classification Pipeline
    print_section_header("Binary Classification Pipeline")
    print("Dataset: Breast Cancer Wisconsin")
    print("Target: Diagnosis (Malignant/Benign)")
    
    print_step(1, "Splitting and scaling binary data")
    split_and_scale_binary_data()
    print("✓ Data preprocessing completed")
    
    print_step(2, "Training binary classification models")
    print("Models to be trained:")
    print("- Logistic Regression")
    print("- Support Vector Machine")
    print("- Random Forest")
    print("- XGBoost")
    run_binary_models()
    print("✓ Model training completed")
    
    print_step(3, "Evaluating binary classification models")
    print("Generating performance metrics and plots...")
    run_binary_test()
    print("✓ Model evaluation completed")
    
    # Multiclass Classification Pipeline
    print_section_header("Multiclass Classification Pipeline")
    print("Dataset: Data Science Salaries")
    print("Target: Experience Level (EN/MI/SE/EX)")
    
    print_step(1, "Splitting and scaling multiclass data")
    split_and_scale_multiclass_data()
    print("✓ Data preprocessing completed")
    
    print_step(2, "Training multiclass classification models")
    print("Models to be trained:")
    print("- Decision Tree")
    print("- K-Nearest Neighbors")
    print("- Random Forest")
    print("- XGBoost")
    run_multiclass_models()
    print("✓ Model training completed")
    
    print_step(3, "Evaluating multiclass classification models")
    print("Generating performance metrics and plots...")
    run_multiclass_test()
    print("✓ Model evaluation completed")
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    print_section_header("Execution Summary")
    print(f"Total execution time: {duration:.2f} seconds")
    print("\nOutput locations:")
    print("Binary Classification:")
    print("  - Data files: binary/*.csv")
    print("  - Model files: binary/*_best_model.pkl")
    print("  - Plots: binary/plots/*.png")
    print("  - CV results: binary/*_cv_results.csv")
    print("\nMulticlass Classification:")
    print("  - Data files: multiclass/*.csv")
    print("  - Model files: multiclass/*_model.pkl")
    print("  - Plots: multiclass/*.png")
    print("  - CV results: multiclass/*_cv_results.csv")
    print("\n✓ All tasks completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "!"*80)
        print("Error occurred during execution:")
        print(str(e))
        print("!"*80)
        raise
