# Quick Reference: Bayesian Optimization

## Quick Start

### Method 1: Run the provided script
```bash
python entrypoint/train_with_bayesian_optimization.py
```

### Method 2: Use the main() function
```python
from src.pipelines.train_pipeline import main

# Bayesian optimization with 50 trials (recommended)
main(use_bayesian_optimization=True, n_trials=50)

# Fast mode (20 trials)
main(use_bayesian_optimization=True, n_trials=20)

# Thorough mode (100 trials)
main(use_bayesian_optimization=True, n_trials=100)
```

### Method 3: Direct function call
```python
from src.pipelines.train_pipeline import bayesian_optimize_models, train_final_model

# Perform Bayesian optimization
results, best_model_name = bayesian_optimize_models(X_train, y_train, n_trials=50)

# Get optimized parameters
optimized_params = results[best_model_name]['params']

# Train final model with optimized hyperparameters
model, y_prob = train_final_model(
    X_train, y_train, X_test, y_test, 
    best_model_name, 
    optimized_params=optimized_params
)
```

## Key Functions

### 1. `bayesian_optimize_models(X, y, n_trials=50, cv_folds=5)`
Performs Bayesian optimization for Random Forest, XGBoost, and Gradient Boosting.

**Input:**
- `X`: Feature matrix
- `y`: Target vector
- `n_trials`: Number of trials (default: 50)
- `cv_folds`: Cross-validation folds (default: 5)

**Output:**
- Dictionary with optimization results for each model
- Best model name

**Example:**
```python
results, best_name = bayesian_optimize_models(X_train, y_train, n_trials=50)
print(f"Best model: {best_name}")
print(f"Best AUC: {results[best_name]['mean_auc']:.4f}")
print(f"Optimized params: {results[best_name]['params']}")
```

### 2. `train_final_model(X_train, y_train, X_test, y_test, model_name, optimized_params=None)`
Trains the final model with optimized hyperparameters.

**Input:**
- `X_train`, `y_train`: Training data
- `X_test`, `y_test`: Test data
- `model_name`: 'XGBoost', 'Random Forest', or 'Gradient Boosting'
- `optimized_params`: Dict of hyperparameters from Bayesian optimization

**Output:**
- Trained model
- Predicted probabilities on test set

**Example:**
```python
model, y_prob = train_final_model(
    X_train, y_train, X_test, y_test,
    'XGBoost',
    optimized_params=results['XGBoost']['params']
)
```

## Hyperparameter Ranges

### Random Forest
| Parameter | Min | Max | Type |
|-----------|-----|-----|------|
| n_estimators | 50 | 300 | int |
| max_depth | 5 | 30 | int |
| min_samples_split | 2 | 20 | int |
| min_samples_leaf | 1 | 10 | int |
| max_features | sqrt, log2 | - | categorical |

### XGBoost
| Parameter | Min | Max | Scale |
|-----------|-----|-----|-------|
| n_estimators | 50 | 300 | linear |
| max_depth | 3 | 15 | linear |
| learning_rate | 0.01 | 0.3 | log |
| subsample | 0.5 | 1.0 | linear |
| colsample_bytree | 0.5 | 1.0 | linear |
| reg_alpha | 0.0 | 10.0 | linear |
| reg_lambda | 0.0 | 10.0 | linear |

### Gradient Boosting
| Parameter | Min | Max | Type |
|-----------|-----|-----|------|
| n_estimators | 50 | 300 | int |
| max_depth | 3 | 15 | int |
| learning_rate | 0.01 | 0.3 | float (log) |
| subsample | 0.5 | 1.0 | float |
| min_samples_split | 2 | 20 | int |
| min_samples_leaf | 1 | 10 | int |

## Timing Guide

| Mode | Trials | Time | Use Case |
|------|--------|------|----------|
| Fast | 20 | 5-10 min | Experimentation |
| Standard | 50 | 20-40 min | Production (recommended) |
| Thorough | 100 | 40-80 min | Final optimization |

## Expected Improvements

Compared to default hyperparameters:
- **Model Performance**: +2-5% AUC improvement
- **Recall**: +3-8% on target class
- **Precision**: Depends on threshold tuning

Example:
```
Default XGBoost:      AUC = 0.82
Bayesian Optimized:   AUC = 0.86  (+4.9%)
```

## What Gets Optimized

### All 3 Models Simultaneously
1. **Random Forest**: 5 hyperparameters (5-6 parameters per model)
2. **XGBoost**: 7 hyperparameters
3. **Gradient Boosting**: 6 hyperparameters

### Automatic Selection
After optimization, the model with the **highest cross-validation AUC** is selected and trained on the full training set.

## Output Example

```
============================================================
🔍 BAYESIAN OPTIMIZATION - HYPERPARAMETER TUNING
============================================================
   Trials: 50 | CV Folds: 5
============================================================

📊 Optimizing Random Forest...
[Progress bar showing 50 trials]
   ✅ Best Trial #28
   Best AUC: 0.8234
   CV AUC: 0.8234 (+/- 0.0156)
   Best Hyperparameters:
      • n_estimators: 157
      • max_depth: 18
      • min_samples_split: 5
      • min_samples_leaf: 2
      • max_features: sqrt

📊 Optimizing XGBoost...
[Progress bar showing 50 trials]
   ✅ Best Trial #31
   Best AUC: 0.8456
   CV AUC: 0.8456 (+/- 0.0132)
   Best Hyperparameters:
      • n_estimators: 203
      • max_depth: 6
      • learning_rate: 0.0847
      • subsample: 0.7234
      • colsample_bytree: 0.8912
      • reg_alpha: 2.1543
      • reg_lambda: 0.8765

📊 Optimizing Gradient Boosting...
[Progress bar showing 50 trials]
   ✅ Best Trial #22
   Best AUC: 0.8234
   CV AUC: 0.8234 (+/- 0.0145)

============================================================
📈 OPTIMIZATION SUMMARY
============================================================

Model                AUC        Std       
----------------------------------------
XGBoost              0.8456     0.0132    
Gradient Boosting    0.8234     0.0145    
Random Forest        0.8156     0.0167    

============================================================
🏆 BEST MODEL: XGBoost
   AUC: 0.8456 (+/- 0.0264)
============================================================

🎯 Training final XGBoost model...
   Using optimized hyperparameters from Bayesian optimization

📊 Test Set Performance:
   AUC-ROC: 0.8523
   Accuracy: 0.8012
   Precision: 0.7845
   Recall: 0.7234
   F1 Score: 0.7526
```

## Customization

### Change Number of Trials
```python
# Fast
main(use_bayesian_optimization=True, n_trials=20)

# Standard
main(use_bayesian_optimization=True, n_trials=50)

# Thorough
main(use_bayesian_optimization=True, n_trials=100)
```

### Change Hyperparameter Ranges
Edit the `objective` functions in `bayesian_optimize_models()`:

```python
# Example: Increase XGBoost learning rate range
'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True)
```

### Change Optimization Metric
In `bayesian_optimize_models()`, change the scoring metric:

```python
# Current: ROC-AUC
scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

# Alternative: Recall (for imbalanced data)
scores = cross_val_score(model, X, y, cv=cv, scoring='recall', n_jobs=-1)

# Alternative: F1 Score
scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
```

## Dependencies

All required packages are already in `requirements.txt`:
- optuna
- scikit-learn
- xgboost
- pandas
- numpy

## Troubleshooting

### Q: Optimization is too slow
A: Reduce trials (use 20 instead of 50) or reduce CV folds (use 3 instead of 5)

### Q: Results are different each time
A: Seeds are fixed, but XGBoost randomness may vary slightly. This is normal.

### Q: How do I know when to stop optimizing?
A: When best AUC plateaus for 10+ consecutive trials, optimization has converged.

### Q: Should I use Bayesian Optimization?
A: Yes, if:
- You want to improve model performance by 2-5%
- You have 20-80 minutes for hyperparameter search
- Reproducibility is important (seeds are fixed)

## See Also

- Full documentation: [bayesian_optimization.md](bayesian_optimization.md)
- Original train script: [train_pipeline.py](../src/pipelines/train_pipeline.py)
- Example script: [train_with_bayesian_optimization.py](train_with_bayesian_optimization.py)
