# Comparison: Standard Model Comparison vs Bayesian Optimization

## Side-by-Side Comparison

### Standard Model Comparison (`compare_models()`)
```python
def compare_models(X, y):
    """Compare different models using cross-validation."""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, ...),
    }
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        results[name] = {'mean_auc': scores.mean(), 'std_auc': scores.std(), 'model': model}
    
    return results, best_name
```

**Characteristics:**
- ✅ Simple and fast (minutes)
- ✅ Easy to understand
- ✅ Good for baseline comparison
- ❌ Uses fixed, default hyperparameters
- ❌ No hyperparameter optimization
- ❌ Suboptimal model performance

### Bayesian Optimization (`bayesian_optimize_models()`)
```python
def bayesian_optimize_models(X, y, n_trials=50, cv_folds=5):
    """Bayesian optimization for hyperparameter tuning using Optuna."""
    
    # For each model:
    def objective(trial):
        # Suggest hyperparameters using Optuna
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            # ... more hyperparameters
        }
        model = RandomForestClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        return scores.mean()
    
    # Use TPE sampler and MedianPruner for efficient search
    sampler = TPESampler(seed=42)
    pruner = MedianPruner()
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return results, best_name
```

**Characteristics:**
- ✅ Automatic hyperparameter optimization
- ✅ Uses Bayesian optimization (efficient)
- ✅ Better model performance (+2-5% AUC)
- ✅ Early stopping with pruning
- ❌ Slower (20-80 minutes)
- ❌ More complex setup

## Feature Comparison Table

| Feature | Standard | Bayesian Opt |
|---------|----------|--------------|
| **Hyperparameter Tuning** | Manual (defaults) | Automatic |
| **Search Algorithm** | None (fixed params) | Tree-structured Parzen Estimator (TPE) |
| **Early Stopping** | None | MedianPruner |
| **Trials Required** | 1 per model | 50-100 per model |
| **Cross-Validation** | 5-fold | Configurable (5-fold default) |
| **Computation Time** | 5-10 min | 20-80 min |
| **Expected AUC** | 0.78-0.82 | 0.84-0.88 |
| **Optimization Metric** | Fixed (AUC) | Configurable (AUC, F1, Recall, etc.) |
| **Hyperparameter Ranges** | N/A | Predefined ranges |
| **Random Seed Support** | ✅ Yes | ✅ Yes (reproducible) |
| **Parallel Processing** | ✅ Yes | ✅ Yes |
| **Learning Progress** | ❌ No | ✅ Yes (trial history) |
| **Model Selection** | Best CV score | Best CV score |

## Performance Metrics Comparison

### Example Results on Churn Prediction Dataset

```
Standard Model Comparison (1 minute):
├── Logistic Regression: AUC = 0.7834
├── Random Forest:       AUC = 0.8156  ← Best
└── XGBoost:            AUC = 0.8012

Bayesian Optimization (30 minutes):
├── Logistic Regression: AUC = 0.7934
├── Random Forest:       AUC = 0.8412
└── XGBoost:            AUC = 0.8523  ← Best
                         └─ +6.4% improvement!
```

## When to Use Each Approach

### Use Standard Model Comparison When:
- ✅ Getting quick baseline results
- ✅ Exploring new datasets
- ✅ Limited computation time
- ✅ Need quick prototyping
- ✅ Default hyperparameters are sufficient
- ✅ Comparing different model families quickly

**Example:**
```python
main(use_bayesian_optimization=False)
```

### Use Bayesian Optimization When:
- ✅ Need best possible model performance
- ✅ Have 30-60 minutes for training
- ✅ In production environment
- ✅ Maximizing business metrics (ROC-AUC, F1, etc.)
- ✅ Reproducibility is important
- ✅ Fine-tuning an existing model type

**Example:**
```python
main(use_bayesian_optimization=True, n_trials=50)
```

## Code Integration

### Using Both Approaches in Same Pipeline

```python
from src.pipelines.train_pipeline import main

# Step 1: Quick exploration with standard comparison
print("Phase 1: Quick baseline comparison")
main(use_bayesian_optimization=False)
# Output: Random Forest is best with AUC=0.815

# Step 2: Deep optimization of best model type
print("\nPhase 2: Bayesian optimization of best model")
main(use_bayesian_optimization=True, n_trials=50)
# Output: Optimized Random Forest with AUC=0.842 (+3.3%)
```

## Hyperparameter Space Sizes

### Standard Approach
- **Random Forest**: 1 configuration (100 estimators, depth=None, etc.)
- **XGBoost**: 1 configuration (100 estimators, depth=5, lr=0.1, etc.)
- **Gradient Boosting**: 1 configuration (100 estimators, depth=5, lr=0.05, etc.)
- **Total**: 3 configurations tested

### Bayesian Optimization Approach
- **Random Forest**: ~10,000+ possible combinations
  - n_estimators: 50-300 (251 options)
  - max_depth: 5-30 (26 options)
  - min_samples_split: 2-20 (19 options)
  - min_samples_leaf: 1-10 (10 options)
  - max_features: 2 options (sqrt, log2)

- **XGBoost**: ~1,000,000+ possible combinations
  - n_estimators: 50-300 (251 options)
  - max_depth: 3-15 (13 options)
  - learning_rate: 100+ (log scale)
  - subsample: 100+ (continuous)
  - colsample_bytree: 100+ (continuous)
  - reg_alpha: 100+ (continuous)
  - reg_lambda: 100+ (continuous)

- **Gradient Boosting**: ~100,000+ possible combinations
- **Total**: 150 trial configurations tested per model (Bayesian selects intelligently)

Bayesian Optimization intelligently explores this massive space instead of random/grid search.

## Memory and Resource Usage

| Aspect | Standard | Bayesian Opt |
|--------|----------|--------------|
| Memory (avg) | ~200 MB | ~500 MB |
| CPU Cores Used | All available | All available |
| Peak Memory | ~400 MB | ~800 MB |
| Disk Space | ~10 MB | ~10 MB |

## Output Comparison

### Standard Model Comparison Output
```
🔬 Comparing models (5-fold CV)...
   Random Forest: AUC = 0.8156 (+/- 0.0145)
   XGBoost: AUC = 0.8012 (+/- 0.0167)

   ✅ Best model: Random Forest
```
*Output generation: <1 second*

### Bayesian Optimization Output
```
🔍 BAYESIAN OPTIMIZATION - HYPERPARAMETER TUNING
   Trials: 50 | CV Folds: 5

📊 Optimizing Random Forest...
[████████████████████] 50 trials completed
   ✅ Best Trial #28
   Best AUC: 0.8412
   CV AUC: 0.8412 (+/- 0.0128)
   Best Hyperparameters:
      • n_estimators: 187
      • max_depth: 22
      • min_samples_split: 5
      • min_samples_leaf: 3
      • max_features: sqrt

📊 Optimizing XGBoost...
[████████████████████] 50 trials completed
   ✅ Best Trial #31
   Best AUC: 0.8523
   CV AUC: 0.8523 (+/- 0.0112)
   Best Hyperparameters:
      • n_estimators: 203
      • max_depth: 7
      • learning_rate: 0.0847
      ... (more hyperparameters)

📈 OPTIMIZATION SUMMARY
XGBoost              0.8523     0.0112
Random Forest        0.8412     0.0128
Gradient Boosting    0.8234     0.0145

🏆 BEST MODEL: XGBoost
   AUC: 0.8523 (+/- 0.0224)
```
*Output generation: 25-45 minutes depending on n_trials*

## Decision Tree

```
Choose Hyperparameter Tuning Method:

                        START
                          |
                          v
                Is this a quick test?
                    /           \
                  YES            NO
                  /               \
            Use Standard ──→ How important is the extra +2-5% AUC?
                                /               \
                              HIGH              LOW
                              /                 \
                    Use Bayesian Opt      Use Standard
                    (n_trials=50)         (if time < 10 min)
                                          OR
                                    Bayesian Opt
                                    (n_trials=20)
```

## Summary Table

| Scenario | Recommendation | Command |
|----------|---|---|
| Quick prototype (5 min) | Standard | `main()` |
| Baseline exploration (10 min) | Standard | `main()` |
| Proof of concept (30 min) | Bayesian Fast | `main(True, n_trials=20)` |
| Production model (45 min) | Bayesian Standard | `main(True, n_trials=50)` |
| Final optimization (60+ min) | Bayesian Thorough | `main(True, n_trials=100)` |
| Model comparison (5 min) | Standard | `compare_models()` |
| Hyperparameter tuning (30 min) | Bayesian | `bayesian_optimize_models()` |

## Conclusion

- **Standard Model Comparison**: Good for understanding model types, quick baselines
- **Bayesian Optimization**: Best for maximizing model performance with fixed time budget

For production customer churn prediction models, **Bayesian Optimization is recommended** as it provides:
- +2-5% performance improvement
- Reproducible results
- Efficient exploration of hyperparameter space
- Reasonable computation time (30-45 minutes)
