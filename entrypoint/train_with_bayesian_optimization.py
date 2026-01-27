"""
Example script demonstrating Bayesian Optimization hyperparameter tuning.

This script shows how to use the Bayesian Optimization approach for
hyperparameter tuning instead of the standard model comparison.

Run this script to train models using Bayesian Optimization:
    python train_with_bayesian_optimization.py

Options:
    - Fast mode (n_trials=20): Quick demonstration
    - Standard mode (n_trials=50): Recommended for production
    - Thorough mode (n_trials=100): Best optimization but slower
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipelines.train_pipeline import main

def main_with_bayesian():
    """
    Train models using Bayesian Optimization.
    
    This approach:
    1. Uses Optuna's Tree-structured Parzen Estimator (TPE) sampler
    2. Implements early stopping with MedianPruner to skip unpromising trials
    3. Optimizes hyperparameters for Random Forest, XGBoost, and Gradient Boosting
    4. Automatically selects the best model based on cross-validation AUC
    5. Trains the final model using the optimized hyperparameters
    """
    
    print("\n" + "="*70)
    print("BAYESIAN OPTIMIZATION FOR HYPERPARAMETER TUNING")
    print("="*70)
    print("""
This script demonstrates Bayesian Optimization using Optuna.

Key Features:
  • Tree-structured Parzen Estimator (TPE) for efficient sampling
  • MedianPruner for early stopping of unpromising trials
  • 5-fold cross-validation for robust evaluation
  • Optimizes 3 models: Random Forest, XGBoost, Gradient Boosting
  
Hyperparameter Ranges:
  
  Random Forest:
    - n_estimators: 50-300
    - max_depth: 5-30
    - min_samples_split: 2-20
    - min_samples_leaf: 1-10
    - max_features: ['sqrt', 'log2']
  
  XGBoost:
    - n_estimators: 50-300
    - max_depth: 3-15
    - learning_rate: 0.01-0.3 (log scale)
    - subsample: 0.5-1.0
    - colsample_bytree: 0.5-1.0
    - reg_alpha: 0.0-10.0
    - reg_lambda: 0.0-10.0
  
  Gradient Boosting:
    - n_estimators: 50-300
    - max_depth: 3-15
    - learning_rate: 0.01-0.3 (log scale)
    - subsample: 0.5-1.0
    - min_samples_split: 2-20
    - min_samples_leaf: 1-10

Benefits vs Standard Model Comparison:
  ✓ Automatic hyperparameter optimization
  ✓ Efficient search through large parameter spaces
  ✓ Early stopping for computational efficiency
  ✓ Better model performance
  ✓ Statistical robustness with cross-validation
""")
    print("="*70 + "\n")
    
    # Run training with Bayesian optimization
    # n_trials=50 is recommended for good results
    # Use n_trials=20 for quick testing, n_trials=100 for thorough optimization
    main(use_bayesian_optimization=True, n_trials=50)


if __name__ == '__main__':
    main_with_bayesian()
