"""
Evaluation Module
Generate comprehensive model evaluation metrics and visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score, precision_score, recall_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')


def load_model_and_data():
    """Load model and test data."""
    project_root = Path(__file__).parent.parent.parent
    
    # Load model
    model = joblib.load(project_root / 'models' / 'churn_model.pkl')
    feature_names = joblib.load(project_root / 'models' / 'feature_names.pkl')
    
    # Load data
    df = pd.read_csv(project_root / 'data' / '02-preprocessed' / 'features.csv')
    
    y = df['Churn'].values
    X = df.drop(columns=['Churn']).values
    
    return model, X, y, feature_names


def plot_roc_curve(y_true, y_prob, save_path):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')
    
    ax.fill_between(fpr, tpr, alpha=0.2, color='#1f77b4')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Churn Prediction Model', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def plot_precision_recall_curve(y_true, y_prob, save_path):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, color='#ff7f0e', lw=2, label=f'PR Curve (AP = {ap:.3f})')
    
    # Add baseline (prevalence)
    baseline = y_true.mean()
    ax.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.2f})')
    
    ax.fill_between(recall, precision, alpha=0.2, color='#ff7f0e')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve - Churn Prediction Model', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'],
                annot_kws={'size': 16})
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def plot_feature_importance(model, feature_names, save_path, top_n=15):
    """Plot feature importance."""
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # For models without feature_importances_, skip
        print("   Model doesn't have feature_importances_ attribute")
        return
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]
    bars = ax.barh(range(top_n), top_importances[::-1], color=colors)
    
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Top Features for Churn Prediction', fontsize=14, fontweight='bold')
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, top_importances[::-1])):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def plot_lift_curve(y_true, y_prob, save_path):
    """Plot cumulative lift curve."""
    # Sort by probability descending
    sorted_idx = np.argsort(y_prob)[::-1]
    y_sorted = y_true[sorted_idx]
    
    # Calculate cumulative lift
    n = len(y_true)
    percentiles = np.arange(1, n+1) / n * 100
    cumulative_churn = np.cumsum(y_sorted) / y_sorted.sum() * 100
    
    # Random baseline
    random_baseline = percentiles
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(percentiles, cumulative_churn, color='#2ca02c', lw=2, label='Model')
    ax.plot(percentiles, random_baseline, color='gray', linestyle='--', label='Random')
    
    ax.fill_between(percentiles, random_baseline, cumulative_churn, 
                     where=(cumulative_churn >= random_baseline),
                     alpha=0.2, color='#2ca02c', label='Lift Area')
    
    ax.set_xlabel('% of Customers Contacted (sorted by risk)', fontsize=12)
    ax.set_ylabel('% of Churners Captured', fontsize=12)
    ax.set_title('Cumulative Gains Chart', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    
    # Add annotations
    for pct in [20, 40, 60]:
        idx = int(n * pct / 100)
        captured = cumulative_churn[idx]
        ax.annotate(f'{pct}% → {captured:.0f}%', 
                   xy=(pct, captured),
                   xytext=(pct+10, captured-10),
                   arrowprops=dict(arrowstyle='->', color='gray'),
                   fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def plot_probability_distribution(y_true, y_prob, save_path):
    """Plot probability distribution by actual class."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate by class
    probs_no_churn = y_prob[y_true == 0]
    probs_churn = y_prob[y_true == 1]
    
    ax.hist(probs_no_churn, bins=50, alpha=0.5, label='No Churn', color='#1f77b4', density=True)
    ax.hist(probs_churn, bins=50, alpha=0.5, label='Churn', color='#ff7f0e', density=True)
    
    ax.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
    
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Probability Distribution by Actual Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def generate_evaluation_report(y_true, y_prob, threshold=0.5):
    """Generate comprehensive evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        'AUC-ROC': roc_auc_score(y_true, y_prob),
        'Average Precision': average_precision_score(y_true, y_prob),
        'Accuracy': (y_pred == y_true).mean(),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
    }
    
    # Lift metrics
    for pct in [10, 20, 30]:
        n_top = int(len(y_true) * pct / 100)
        sorted_idx = np.argsort(y_prob)[::-1]
        top_churners = y_true[sorted_idx][:n_top].sum()
        total_churners = y_true.sum()
        metrics[f'Recall @{pct}%'] = top_churners / total_churners
        metrics[f'Lift @{pct}%'] = (top_churners / n_top) / y_true.mean()
    
    return metrics


def main():
    """Generate all evaluation metrics and plots."""
    print("=" * 60)
    print("CUSTOMER CHURN - MODEL EVALUATION")
    print("=" * 60)
    
    # Setup
    project_root = Path(__file__).parent.parent
    img_dir = project_root / 'docs' / 'img'
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    print("\n📂 Loading model and data...")
    model, X, y, feature_names = load_model_and_data()
    
    # Get predictions
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Generate metrics
    print("\n📊 Evaluation Metrics:")
    metrics = generate_evaluation_report(y, y_prob)
    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")
    
    # Generate plots
    print("\n📈 Generating visualizations...")
    plot_roc_curve(y, y_prob, img_dir / 'roc_curve.png')
    plot_precision_recall_curve(y, y_prob, img_dir / 'pr_curve.png')
    plot_confusion_matrix(y, y_pred, img_dir / 'confusion_matrix.png')
    plot_feature_importance(model, feature_names, img_dir / 'feature_importance.png')
    plot_lift_curve(y, y_prob, img_dir / 'lift_curve.png')
    plot_probability_distribution(y, y_prob, img_dir / 'probability_distribution.png')
    
    # Create combined performance chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ROC
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc = roc_auc_score(y, y_prob)
    axes[0, 0].plot(fpr, tpr, color='#1f77b4', lw=2, label=f'AUC = {auc:.3f}')
    axes[0, 0].plot([0, 1], [0, 1], 'k--')
    axes[0, 0].fill_between(fpr, tpr, alpha=0.2)
    axes[0, 0].set_title('ROC Curve', fontweight='bold')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].legend()
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    axes[0, 1].set_title('Confusion Matrix', fontweight='bold')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_n = 10
        indices = np.argsort(importances)[::-1][:top_n]
        axes[1, 0].barh(range(top_n), importances[indices][::-1])
        axes[1, 0].set_yticks(range(top_n))
        axes[1, 0].set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=8)
        axes[1, 0].set_title('Top 10 Features', fontweight='bold')
        axes[1, 0].set_xlabel('Importance')
    
    # Probability Distribution
    axes[1, 1].hist(y_prob[y==0], bins=30, alpha=0.5, label='No Churn', density=True)
    axes[1, 1].hist(y_prob[y==1], bins=30, alpha=0.5, label='Churn', density=True)
    axes[1, 1].axvline(x=0.5, color='red', linestyle='--')
    axes[1, 1].set_title('Score Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(img_dir / 'model_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {img_dir / 'model_performance.png'}")
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
