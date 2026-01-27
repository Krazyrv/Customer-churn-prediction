"""
Model Training Module
Train and save churn prediction model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
    print("✅ XGBoost is available.")
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not installed. Using GradientBoosting instead.")


def load_processed_data():
    """Load preprocessed data."""
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / 'data' / 'processed'
    
    # Load features
    df = pd.read_csv(processed_dir / 'features.csv')
    
    # Separate features and target
    y = df['Churn'].values
    X = df.drop(columns=['Churn']).values
    
    # Load feature names
    with open(processed_dir / 'feature_names.txt', 'r') as f:
        feature_names = f.read().strip().split('\n')
    
    return X, y, feature_names



def compare_models(X, y):
    """Compare different models using cross-validation."""
    print("\n🔬 Comparing models (5-fold CV)...")
    models = {
        # 'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        # 'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    if HAS_XGBOOST:
        print("Add XGBoost")
        models['XGBoost'] = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        # scores = cross_val_score(model, X, y, cv=cv, scoring='recall', n_jobs=-1)
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        results[name] = {
            'mean_auc': scores.mean(),
            'std_auc': scores.std(),
            'model': model
        }
        print(f"   {name}: AUC = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    # Find best model
    best_name = max(results, key=lambda x: results[x]['mean_auc'])
    print(f"\n   ✅ Best model: {best_name}")
    
    return results, best_name


def bayesian_optimize_models(X, y, n_trials=50, cv_folds=5):
    """
    Bayesian optimization for hyperparameter tuning using Optuna.
    
    This function uses Optuna to perform Bayesian optimization (Tree-structured Parzen 
    Estimator) to find the best hyperparameters for multiple models.
    
    Args:
        X: Feature matrix
        y: Target vector
        n_trials: Number of optimization trials (default: 50)
        cv_folds: Number of cross-validation folds (default: 5)
    
    Returns:
        tuple: (best_model_results_dict, best_model_name)
    """
    print("\n" + "="*60)
    print("🔍 BAYESIAN OPTIMIZATION - HYPERPARAMETER TUNING")
    print("="*60)
    print(f"   Trials: {n_trials} | CV Folds: {cv_folds}")
    print("="*60)
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}
    
    # ==================== Random Forest Optimization ====================
    print("\n📊 Optimizing Random Forest...")
    
    def rf_objective(trial):
        """Objective function for Random Forest optimization."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()
    
    sampler_rf = TPESampler(seed=42)
    pruner_rf = MedianPruner()
    study_rf = optuna.create_study(sampler=sampler_rf, pruner=pruner_rf, direction='maximize')
    study_rf.optimize(rf_objective, n_trials=n_trials, show_progress_bar=True)
    
    best_rf_params = study_rf.best_params
    best_rf_model = RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(best_rf_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    results['Random Forest'] = {
        'mean_auc': rf_scores.mean(),
        'std_auc': rf_scores.std(),
        'model': best_rf_model,
        'params': best_rf_params,
        'best_trial': study_rf.best_trial.number,
        'best_value': study_rf.best_value
    }
    
    print(f"\n   ✅ Best Trial #{study_rf.best_trial.number}")
    print(f"   Best AUC: {study_rf.best_value:.4f}")
    print(f"   CV AUC: {rf_scores.mean():.4f} (+/- {rf_scores.std()*2:.4f})")
    print(f"   Best Hyperparameters:")
    for key, value in best_rf_params.items():
        print(f"      • {key}: {value}")
    
    # ==================== XGBoost Optimization ====================
    if HAS_XGBOOST:
        print("\n📊 Optimizing XGBoost...")
        
        def xgb_objective(trial):
            """Objective function for XGBoost optimization."""
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'verbosity': 0
            }
            
            model = XGBClassifier(**params)
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
            return scores.mean()
        
        sampler_xgb = TPESampler(seed=42)
        pruner_xgb = MedianPruner()
        study_xgb = optuna.create_study(sampler=sampler_xgb, pruner=pruner_xgb, direction='maximize')
        study_xgb.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=True)
        
        best_xgb_params = study_xgb.best_params
        best_xgb_model = XGBClassifier(
            **best_xgb_params, 
            random_state=42, 
            use_label_encoder=False, 
            eval_metric='logloss',
            verbosity=0
        )
        xgb_scores = cross_val_score(best_xgb_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        results['XGBoost'] = {
            'mean_auc': xgb_scores.mean(),
            'std_auc': xgb_scores.std(),
            'model': best_xgb_model,
            'params': best_xgb_params,
            'best_trial': study_xgb.best_trial.number,
            'best_value': study_xgb.best_value
        }
        
        print(f"\n   ✅ Best Trial #{study_xgb.best_trial.number}")
        print(f"   Best AUC: {study_xgb.best_value:.4f}")
        print(f"   CV AUC: {xgb_scores.mean():.4f} (+/- {xgb_scores.std()*2:.4f})")
        print(f"   Best Hyperparameters:")
        for key, value in best_xgb_params.items():
            print(f"      • {key}: {value}")
    
    # ==================== Gradient Boosting Optimization ====================
    print("\n📊 Optimizing Gradient Boosting...")
    
    def gb_objective(trial):
        """Objective function for Gradient Boosting optimization."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': 42
        }
        
        model = GradientBoostingClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()
    
    sampler_gb = TPESampler(seed=42)
    pruner_gb = MedianPruner()
    study_gb = optuna.create_study(sampler=sampler_gb, pruner=pruner_gb, direction='maximize')
    study_gb.optimize(gb_objective, n_trials=n_trials, show_progress_bar=True)
    
    best_gb_params = study_gb.best_params
    best_gb_model = GradientBoostingClassifier(**best_gb_params, random_state=42)
    gb_scores = cross_val_score(best_gb_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    results['Gradient Boosting'] = {
        'mean_auc': gb_scores.mean(),
        'std_auc': gb_scores.std(),
        'model': best_gb_model,
        'params': best_gb_params,
        'best_trial': study_gb.best_trial.number,
        'best_value': study_gb.best_value
    }
    
    print(f"\n   ✅ Best Trial #{study_gb.best_trial.number}")
    print(f"   Best AUC: {study_gb.best_value:.4f}")
    print(f"   CV AUC: {gb_scores.mean():.4f} (+/- {gb_scores.std()*2:.4f})")
    print(f"   Best Hyperparameters:")
    for key, value in best_gb_params.items():
        print(f"      • {key}: {value}")
    
    # ==================== Summary and Best Model ====================
    print("\n" + "="*60)
    print("📈 OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"\n{'Model':<20} {'AUC':<10} {'Std':<10}")
    print("-" * 40)
    
    for name in sorted(results.keys(), key=lambda x: results[x]['mean_auc'], reverse=True):
        print(f"{name:<20} {results[name]['mean_auc']:<10.4f} {results[name]['std_auc']:<10.4f}")
    
    # Find best model
    best_name = max(results, key=lambda x: results[x]['mean_auc'])
    print("\n" + "="*60)
    print(f"🏆 BEST MODEL: {best_name}")
    print(f"   AUC: {results[best_name]['mean_auc']:.4f} (+/- {results[best_name]['std_auc']*2:.4f})")
    print("="*60)
    
    return results, best_name


def train_final_model(X_train, y_train, X_test, y_test, model_name = 'XGBoost', optimized_params=None):
    """Train the final model with optimized hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model to train
        optimized_params: Optional dict of optimized hyperparameters from Bayesian optimization
    """
    print(f"\n🎯 Training final {model_name} model...")
    
    if model_name == 'XGBoost' and HAS_XGBOOST:
        if optimized_params:
            # Use optimized parameters from Bayesian optimization
            model = XGBClassifier(
                **optimized_params,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
            print("   Using optimized hyperparameters from Bayesian optimization")
        else:
            # Use default parameters
            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
    elif model_name == 'Random Forest':
        if optimized_params:
            # Use optimized parameters from Bayesian optimization
            model = RandomForestClassifier(
                **optimized_params,
                random_state=42,
                n_jobs=-1
            )
            print("   Using optimized hyperparameters from Bayesian optimization")
        else:
            # Use default parameters
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
    elif model_name == 'Gradient Boosting':
        if optimized_params:
            # Use optimized parameters from Bayesian optimization
            model = GradientBoostingClassifier(
                **optimized_params,
                random_state=42
            )
            print("   Using optimized hyperparameters from Bayesian optimization")
        else:
            # Use default parameters
            model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)  

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n📊 Test Set Performance:")
    print(f"   AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"   Accuracy: {(y_pred == y_test).mean():.4f}")
    print(f"   Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"   Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"   F1 Score: {f1_score(y_test, y_pred):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 No    Yes")
    print(f"   Actual No    {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"          Yes   {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    return model, y_prob

def calculate_lift(y_true, y_prob, percentile=20):
    """Calculate lift at given percentile."""
    # Sort by probability
    sort_idx = np.argsort(y_prob)[::-1]
    y_sorted = y_true[sort_idx]

    # Get top percentiles
    n_top = int(len(y_true) * percentile / 100)
    top_churn_rate = y_sorted[:n_top].mean()
    base_churn_rate = y_true.mean()

    lift = top_churn_rate / base_churn_rate
    print(f"\n📈 Lift at top {percentile}%: {lift:.2}")

    return lift, top_churn_rate, base_churn_rate

def analyze_thresholds(y_test, y_prob):
    """Analyze different classification thresholds."""
    print("\n📈 Threshold Analysis:")
    print(f"   {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("   " + "-" * 48)
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = (y_prob >= threshold).astype(int)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"   {threshold:<12.1f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")


def save_model(model, feature_names):
    """Save model and metadata."""
    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / 'churn_model.pkl'
    joblib.dump(model, model_path)
    
    # Save feature names
    feature_path = model_dir / 'feature_names.pkl'
    joblib.dump(feature_names, feature_path)
    
    print(f"\n💾 Model saved to {model_path}")
    
    return model_path

def main(use_bayesian_optimization=False, n_trials=50):
    """Main training pipeline.
    
    Args:
        use_bayesian_optimization: If True, use Bayesian optimization instead of model comparison
        n_trials: Number of trials for Bayesian optimization (only used if use_bayesian_optimization=True)
    """
    print("=" * 60)
    print("CUSTOMER CHURN - MODEL TRAINING")
    print("=" * 60)
    
    # Try to load processed data, or run preprocessing
    try:
        X, y, feature_names = load_processed_data()
        print(f"✅ Loaded processed data: {X.shape[0]} samples, {X.shape[1]} features")
    except FileNotFoundError:
        print("⚠️ Processed data not found. Running preprocessing...")
        from pipelines.preprocess_pipeline import prepare_data
        # from preprocess_pipeline import prepare_data
        X_train, X_test, y_train, y_test, feature_names = prepare_data()
        # Combine for comparison
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, y_test])


    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Choose hyperparameter tuning approach
    if use_bayesian_optimization:
        print("\n🚀 Using BAYESIAN OPTIMIZATION for hyperparameter tuning")
        results, best_name = bayesian_optimize_models(X_train, y_train, n_trials=n_trials)
        # Extract optimized parameters for final training
        optimized_params = results[best_name]['params']
    else:
        print("\n🚀 Using STANDARD MODEL COMPARISON")
        results, best_name = compare_models(X_train, y_train)
        optimized_params = None
    
    print(f"\n🏆 Selected model for final training: {best_name}")

    # Train final model with optimized parameters
    model, y_prob = train_final_model(
        X_train, y_train, X_test, y_test, 
        best_name, 
        optimized_params=optimized_params
    )
    
    # Calculate lift
    lift, top_rate, base_rate = calculate_lift(y_test, y_prob, percentile=20)
    print(f"\n📊 Lift Analysis:")
    print(f"   Base churn rate: {base_rate*100:.1f}%")
    print(f"   Top 20% churn rate: {top_rate*100:.1f}%")
    print(f"   Lift: {lift:.2f}x")
    
    # Threshold analysis
    analyze_thresholds(y_test, y_prob)
    
    # Save model
    save_model(model, feature_names)
    
    print("\n✅ Training complete!")

if __name__ == '__main__':
    main()
