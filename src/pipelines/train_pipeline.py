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


def train_final_model(X_train, y_train, X_test, y_test, model_name = 'XGBoost'):
    """Train the final model with optimized hyperparameters."""
    print(f"\n🎯 Training final {model_name} model...")
    if model_name == 'XGBoost' and HAS_XGBOOST:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
    elif model_name == 'Gradient Boosting':
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

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("CUSTOMER CHURN - MODEL TRAINING")
    print("=" * 60)
    
    # Try to load processed data, or run preprocessing
    try:
        X, y, feature_names = load_processed_data()
        print(f"✅ Loaded processed data: {X.shape[0]} samples, {X.shape[1]} features")
    except FileNotFoundError:
        print("⚠️ Processed data not found. Running preprocessing...")
        # from pipelines.preprocess_pipeline import prepare_data
        from preprocess_pipeline import prepare_data
        X_train, X_test, y_train, y_test, feature_names = prepare_data()
        # Combine for comparison
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, y_test])


    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compare models
    results, best_name = compare_models(X_train, y_train)
    
    print(f"\n🏆 Selected model for final training: {best_name}")

    # Train final model
    model, y_prob = train_final_model(X_train, y_train, X_test, y_test, best_name)
    
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
