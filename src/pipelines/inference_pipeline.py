"""
Prediction Module
Load trained model and make predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib


def load_model():
    """Load the trained churn prediction model."""
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / 'models' / 'churn_model.pkl'
    feature_name = project_root / 'models' / 'feature_names.pkl'
    
    model = joblib.load(model_path)
    feature_names = joblib.load(feature_name)
    return model, feature_names


def get_recommendation(probability: float) -> str:
    """Get retention recommendation based on churn probability."""
    if probability >= 0.7:
        return "URGENT: Immediate intervention required. Offer significant discount or free upgrade."
    elif probability >= 0.5:
        return "HIGH PRIORITY: Schedule retention call. Consider contract upgrade offer."
    elif probability >= 0.3:
        return "MONITOR: Add to watch list. Send satisfaction survey."
    else:
        return "LOW RISK: Standard engagement. Consider upsell opportunities."
    



def predict_single(customer_data: dict, model = None, feature_names = None):
    """Make a prediction for a single customer."""
    if model is None or feature_names is None:
        model, feature_names = load_model()


    features = pd.DataFrame([customer_data])

    # Ensure all features are present
    for feat in feature_names:
        if feat not in features.columns:
            features[feat] = 0

    features = features[feature_names]

    
    prob = model.predict_proba(features.values)[0, 1]
    pred = 'High Risk' if prob >= 0.5 else 'Low Risk'


    return {
        'churn_probability': prob,
        'risk_category': pred,
        'recommendation': get_recommendation(prob)
    }


def get_top_risk_customers(df: pd.DataFrame, n: int = 100, model=None, feature_names=None):
    """Get top N highest risk customers."""
    result = predict_batch(df, model, feature_names)
    return result.nlargest(n, 'churn_probability')


def predict_batch(df: pd.DataFrame, model=None, feature_names=None):
    """Make predictions for a batch of customers."""
    if model is None or feature_names is None:
        model, feature_names = load_model()


    # Ensure all features are present
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0

    # Order features correctly
    X = df[feature_names].values
    
    # Predict
    probs = model.predict_proba(X)[:, 1]

    result = df.copy()
    result['churn_probability'] = probs
    result['risk_category'] = np.where(probs >= 0.5, 'High Risk', 'Low Risk')
    result['risk_score'] = pd.qcut(probs, 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    preds = np.where(probs >= 0.5, 'High Risk', 'Low Risk')

    results = pd.DataFrame({
        'customer_id': df['customer_id'],
        'churn_probability': probs,
        'risk_category': preds
    })

    # results['recommendation'] = results['churn_probability'].apply(get_recommendation)

    return results


def explain_prediction(customer_data: dict, model=None, feature_names=None):
    """
    Explain prediction for a single customer using feature contributions.
    
    Returns top factors increasing and decreasing churn risk.
    """
    if model is None:
        model, feature_names = load_model()
    
    # Try to use SHAP if available
    try:
        import shap
        
        # Create feature vector
        features = pd.DataFrame([customer_data])
        for feat in feature_names:
            if feat not in features.columns:
                features[feat] = 0
        features = features[feature_names]
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features.values)
        
        # Get feature contributions
        contributions = dict(zip(feature_names, shap_values[0]))
        
        # Sort by absolute contribution
        sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Top factors increasing risk (positive SHAP)
        increasing = [(k, v) for k, v in sorted_contrib if v > 0][:5]
        
        # Top factors decreasing risk (negative SHAP)
        decreasing = [(k, v) for k, v in sorted_contrib if v < 0][:5]
        
        return {
            'increasing_risk': increasing,
            'decreasing_risk': decreasing
        }
        
    except ImportError:
        return {
            'increasing_risk': [],
            'decreasing_risk': [],
            'note': 'Install SHAP for feature explanations: pip install shap'
        }
    

def main():
    """Example usage."""
    print("Loading model...")
    model, feature_names = load_model()
    print(f"Model loaded with {len(feature_names)} features")
    
    # Example prediction
    example_customer = {
        # High-risk customer example
        'tenure': 12,
        'MonthlyCharges': 75.0,
        'TotalCharges': 900.0,
        'Contract_Month-to-month': 1,
        'Contract_One year': 0,
        'Contract_Two year': 0,
        'InternetService_Fiber optic': 1,
        'InternetService_DSL': 0,
        'InternetService_No': 0,
        'TechSupport_No': 1,
        'TechSupport_Yes': 0,
        'OnlineSecurity_No': 1,
        'OnlineSecurity_Yes': 0,
    }

    # example_customer = {
    #     # Low-risk customer example
    #     'tenure': 22,
    #     'MonthlyCharges': 35.0,
    #     'TotalCharges': 1900.0,
    #     'Contract_Month-to-month': 0,
    #     'Contract_One year': 1,
    #     'Contract_Two year': 0,
    #     'InternetService_Fiber optic': 1,
    #     'InternetService_DSL': 1,
    #     'InternetService_No': 1,
    #     'TechSupport_No': 1,
    #     'TechSupport_Yes': 1,
    #     'OnlineSecurity_No': 1,
    #     'OnlineSecurity_Yes': 0,
    # }

    
    # Fill in remaining features with 0
    for feat in feature_names:
        if feat not in example_customer:
            example_customer[feat] = 0
    
    result = predict_single(example_customer, model, feature_names)
    print(f"\nExample Prediction:")
    print(f"   Churn Probability: {result['churn_probability']*100:.1f}%")
    print(f"   Risk Category: {result['risk_category']}")
    print(f"   Recommendation: {result['recommendation']}")
    print(f"   Prediction Explaination: {explain_prediction(example_customer)}")

if __name__ == '__main__':
    main()