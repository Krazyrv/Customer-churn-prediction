"""
Churn Prediction Dashboard
Interactive app for predicting customer churn risk.

Run: streamlit run demo/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Page config
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .risk-high { background-color: #ffcdd2; padding: 20px; border-radius: 10px; }
    .risk-medium { background-color: #fff9c4; padding: 20px; border-radius: 10px; }
    .risk-low { background-color: #c8e6c9; padding: 20px; border-radius: 10px; }
    .metric-box { background-color: #f0f2f6; padding: 15px; border-radius: 5px; text-align: center; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model."""
    project_root = Path(__file__).parent.parent
    model_dir = project_root / 'models'
    
    try:
        model = joblib.load(model_dir / 'churn_model.pkl')
        feature_names = joblib.load(model_dir / 'feature_names.pkl')
        return model, feature_names
    except FileNotFoundError:
        st.error("⚠️ Model not found. Please train the model first: `python src/train_pipeline.py`")
        st.stop()


@st.cache_data
def load_data():
    """Load processed data for analysis."""
    project_root = Path(__file__).parent.parent#.parent.parent
    # Test
    # For local use
    # project_root = Path(__file__).parent.parent
    
    try:
        df = pd.read_csv(project_root / 'data' / '02-preprocessed' / 'processed_data.csv')
        return df
    except FileNotFoundError:
        # return None, f"{project_root} / 'data' / '02-preprocessed' / 'processed_data.csv'"
        return None#, f"{project_root} / 'data' / '02-preprocessed' / 'processed_data.csv'"


def create_feature_vector(inputs, feature_names):
    """Create feature vector from user inputs."""
    # Start with zeros
    features = {name: 0 for name in feature_names}
    
    # Fill in numeric features
    features['tenure'] = inputs['tenure']
    features['MonthlyCharges'] = inputs['monthly_charges']
    features['TotalCharges'] = inputs['total_charges']
    
    # Engineered features
    features['avg_monthly_charges'] = inputs['monthly_charges'] if inputs['tenure'] == 0 else inputs['total_charges'] / inputs['tenure']
    features['charges_tenure_ratio'] = inputs['monthly_charges'] / (inputs['tenure'] + 1)
    features['is_new_customer'] = 1 if inputs['tenure'] <= 6 else 0
    
    # Contract risk
    contract_risk_map = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
    features['contract_risk'] = contract_risk_map.get(inputs['contract'], 2)
    
    # Payment risk
    payment_risk_map = {
        'Electronic check': 3,
        'Mailed check': 2,
        'Bank transfer (automatic)': 1,
        'Credit card (automatic)': 1
    }
    features['payment_risk'] = payment_risk_map.get(inputs['payment_method'], 2)
    
    # Service counts
    services = [inputs['phone_service'], inputs['internet_service'] != 'No',
                inputs['online_security'], inputs['online_backup'],
                inputs['device_protection'], inputs['tech_support'],
                inputs['streaming_tv'], inputs['streaming_movies']]
    features['num_services'] = sum(services)
    
    features['has_streaming'] = 1 if inputs['streaming_tv'] or inputs['streaming_movies'] else 0
    features['has_security_support'] = 1 if (inputs['online_security'] or inputs['tech_support'] or 
                                              inputs['online_backup'] or inputs['device_protection']) else 0
    
    # One-hot encode categorical features
    # Gender
    if f"gender_{inputs['gender']}" in feature_names:
        features[f"gender_{inputs['gender']}"] = 1
    
    # Senior citizen
    if f"SeniorCitizen_{inputs['senior_citizen']}" in feature_names:
        features[f"SeniorCitizen_{inputs['senior_citizen']}"] = 1
    
    # Partner
    if f"Partner_{'Yes' if inputs['partner'] else 'No'}" in feature_names:
        features[f"Partner_{'Yes' if inputs['partner'] else 'No'}"] = 1
    
    # Dependents
    if f"Dependents_{'Yes' if inputs['dependents'] else 'No'}" in feature_names:
        features[f"Dependents_{'Yes' if inputs['dependents'] else 'No'}"] = 1
    
    # Contract
    if f"Contract_{inputs['contract']}" in feature_names:
        features[f"Contract_{inputs['contract']}"] = 1
    
    # Internet Service
    if f"InternetService_{inputs['internet_service']}" in feature_names:
        features[f"InternetService_{inputs['internet_service']}"] = 1
    
    # Payment Method
    if f"PaymentMethod_{inputs['payment_method']}" in feature_names:
        features[f"PaymentMethod_{inputs['payment_method']}"] = 1
    
    # Tech Support
    tech_val = 'Yes' if inputs['tech_support'] else 'No'
    if f"TechSupport_{tech_val}" in feature_names:
        features[f"TechSupport_{tech_val}"] = 1
    
    # Online Security
    sec_val = 'Yes' if inputs['online_security'] else 'No'
    if f"OnlineSecurity_{sec_val}" in feature_names:
        features[f"OnlineSecurity_{sec_val}"] = 1
    
    # Paperless Billing
    if f"PaperlessBilling_{'Yes' if inputs['paperless_billing'] else 'No'}" in feature_names:
        features[f"PaperlessBilling_{'Yes' if inputs['paperless_billing'] else 'No'}"] = 1
    
    return pd.DataFrame([features])[feature_names]


def get_risk_factors(inputs):
    """Identify risk factors for the customer."""
    risk_factors = []
    protective_factors = []
    
    # Tenure
    if inputs['tenure'] < 12:
        risk_factors.append("New customer (< 12 months)")
    elif inputs['tenure'] > 48:
        protective_factors.append("Long-term customer (4+ years)")
    
    # Contract
    if inputs['contract'] == 'Month-to-month':
        risk_factors.append("Month-to-month contract (high churn)")
    elif inputs['contract'] == 'Two year':
        protective_factors.append("2-year contract (committed)")
    
    # Services
    if not inputs['tech_support'] and inputs['internet_service'] != 'No':
        risk_factors.append("No tech support")
    if not inputs['online_security'] and inputs['internet_service'] != 'No':
        risk_factors.append("No online security")
    
    # Internet
    if inputs['internet_service'] == 'Fiber optic':
        risk_factors.append("Fiber optic service (higher churn rate)")
    
    # Payment
    if inputs['payment_method'] == 'Electronic check':
        risk_factors.append("Electronic check payment (highest churn)")
    
    # Charges
    if inputs['monthly_charges'] > 80:
        risk_factors.append("High monthly charges (>$80)")
    
    return risk_factors, protective_factors


def render_prediction_result(probability, risk_factors, protective_factors):
    """Render the prediction result with styling."""
    
    # Determine risk level
    if probability >= 0.7:
        risk_level = 'HIGH'
        risk_class = 'risk-high'
        emoji = '🔴'
    elif probability >= 0.4:
        risk_level = 'MEDIUM'
        risk_class = 'risk-medium'
        emoji = '🟡'
    else:
        risk_level = 'LOW'
        risk_class = 'risk-low'
        emoji = '🟢'
    
    st.markdown(f"""
    <div class="{risk_class}">
        <h2 style="margin:0;">{emoji} Churn Risk: {risk_level}</h2>
        <h3 style="margin:10px 0;">Probability: {probability*100:.1f}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚠️ Risk Factors")
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"• {factor}")
        else:
            st.markdown("*No significant risk factors identified*")
    
    with col2:
        st.markdown("#### ✅ Protective Factors")
        if protective_factors:
            for factor in protective_factors:
                st.markdown(f"• {factor}")
        else:
            st.markdown("*No significant protective factors identified*")


def render_recommendations(probability, inputs):
    """Render retention recommendations."""
    st.markdown("### 💡 Recommended Actions")
    
    recommendations = []
    
    if inputs['contract'] == 'Month-to-month' and probability >= 0.3:
        recommendations.append({
            'action': 'Offer Contract Upgrade',
            'details': 'Offer 20% discount for switching to annual contract',
            'impact': 'High',
            'cost': '$50-100'
        })
    
    if not inputs['tech_support'] and inputs['internet_service'] != 'No':
        recommendations.append({
            'action': 'Bundle Tech Support',
            'details': 'Offer free tech support for 3 months',
            'impact': 'Medium',
            'cost': '$30'
        })
    
    if inputs['tenure'] < 12:
        recommendations.append({
            'action': 'Onboarding Check-in',
            'details': 'Schedule personal check-in call',
            'impact': 'Medium',
            'cost': '$10'
        })
    
    if probability >= 0.5:
        recommendations.append({
            'action': 'Retention Offer',
            'details': 'Proactive discount or service upgrade',
            'impact': 'High',
            'cost': '$50-150'
        })
    
    if probability >= 0.7:
        recommendations.append({
            'action': 'Urgent Intervention',
            'details': 'Manager callback within 24 hours',
            'impact': 'Very High',
            'cost': '$20'
        })
    
    if recommendations:
        for rec in recommendations:
            with st.expander(f"**{rec['action']}** (Impact: {rec['impact']})"):
                st.write(f"**Details:** {rec['details']}")
                st.write(f"**Estimated Cost:** {rec['cost']}")
    else:
        st.info("This customer appears to be low risk. Focus on upselling opportunities.")


def render_single_prediction():
    """Render the single customer prediction interface."""
    st.header("🔮 Predict Customer Churn")
    
    model, feature_names = load_model()
    
    # Input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ['Male', 'Female'])
        senior_citizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
        partner = st.checkbox("Has Partner")
        dependents = st.checkbox("Has Dependents")
    
    with col2:
        st.subheader("Account Info")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.checkbox("Paperless Billing", value=True)
        payment_method = st.selectbox("Payment Method", [
            'Electronic check', 'Mailed check', 
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
    
    with col3:
        st.subheader("Charges")
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges))
    
    st.subheader("Services")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        phone_service = st.checkbox("Phone Service", value=True)
        internet_service = st.selectbox("Internet Service", ['No', 'DSL', 'Fiber optic'])
    
    with col2:
        online_security = st.checkbox("Online Security")
        online_backup = st.checkbox("Online Backup")
        device_protection = st.checkbox("Device Protection")
    
    with col3:
        tech_support = st.checkbox("Tech Support")
        streaming_tv = st.checkbox("Streaming TV")
        streaming_movies = st.checkbox("Streaming Movies")
    
    # Predict button
    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        # Collect inputs
        inputs = {
            'gender': gender,
            'senior_citizen': senior_citizen,
            'partner': partner,
            'dependents': dependents,
            'tenure': tenure,
            'contract': contract,
            'paperless_billing': paperless_billing,
            'payment_method': payment_method,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'phone_service': phone_service,
            'internet_service': internet_service,
            'online_security': online_security,
            'online_backup': online_backup,
            'device_protection': device_protection,
            'tech_support': tech_support,
            'streaming_tv': streaming_tv,
            'streaming_movies': streaming_movies,
        }
        
        # Create feature vector
        X = create_feature_vector(inputs, feature_names)
        
        # Predict
        probability = model.predict_proba(X.values)[0, 1]
        
        # Get risk factors
        risk_factors, protective_factors = get_risk_factors(inputs)
        
        st.markdown("---")
        
        # Show results
        render_prediction_result(probability, risk_factors, protective_factors)
        
        st.markdown("---")
        
        # Show recommendations
        render_recommendations(probability, inputs)


def render_batch_analysis():
    """Render batch analysis of customer base."""
    st.header("📊 Customer Base Analysis")
    
    df = load_data()
    # df,txt = load_data()
    
    if df is None:
        st.error("Data not found. Please run preprocessing first.\n Path located at: "+txt)
        return
    
    model, feature_names = load_model()
    
    # Get predictions for all customers
    features_df = pd.read_csv(Path(__file__).parent.parent / 'data' / '02-preprocessed' / 'features.csv')
    X = features_df.drop(columns=['Churn']).values
    y_prob = model.predict_proba(X)[:, 1]
    
    df['Churn_Probability'] = y_prob
    df['Risk_Category'] = pd.cut(y_prob, bins=[0, 0.3, 0.5, 0.7, 1.0], 
                                  labels=['Low', 'Medium', 'High', 'Critical'])
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        st.metric("Avg Churn Risk", f"{y_prob.mean()*100:.1f}%")
    with col3:
        high_risk = (y_prob >= 0.5).sum()
        st.metric("High Risk Customers (> 50% churn probability)", f"{high_risk:,}")
    with col4:
        actual_churn = df['Churn'].sum()
        st.metric("Actual Churners", f"{actual_churn:,}")

    st.header("💰 Projected Annual Revenue Loss")
    # Projected Revenue Loss
    avg_revenue_per_customer = df['MonthlyCharges'].mean() * 12  # Annualized
    projected_loss = (high_risk) * avg_revenue_per_customer
    # st.markdown(f"### 💰 Projected Annual Revenue Loss due to Churn from High-Risk Customers: ${projected_loss:,.0f}")
    # st.markdown(f"### 💰 Probability: ${high_risk}")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Estimate Total Revenue", f"${avg_revenue_per_customer*len(df):,}")
    with col2:
        st.metric("Avg Revenue per Customer", f"${avg_revenue_per_customer:,.0f}")
    with col3:
        high_risk = (y_prob >= 0.5).sum()
        st.metric("Projected Revenue Loss", f"${projected_loss:,.0f}")
    with col4:
        actual_churn = df['Churn'].sum()
        st.metric("Actual Revenue Loss", f"${avg_revenue_per_customer*actual_churn:,.0f}")

    # Risk distribution
    st.subheader("Risk Distribution")
    
    fig = px.histogram(df, x='Churn_Probability', nbins=50, 
                       color='Churn', barmode='overlay',
                       labels={'Churn_Probability': 'Predicted Churn Probability', 'Churn': 'Actually Churned'},
                       color_discrete_map={0: '#1f77b4', 1: '#ff7f0e'})
    fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk by segment
    st.subheader("Risk by Segment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_by_contract = df.groupby('Contract')['Churn_Probability'].mean().reset_index()
        fig = px.bar(risk_by_contract, x='Contract', y='Churn_Probability',
                     title='Average Risk by Contract Type',
                     labels={'Churn_Probability': 'Avg Churn Probability'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        risk_by_tenure = df.groupby('tenure_bucket')['Churn_Probability'].mean().reset_index()
        fig = px.bar(risk_by_tenure, x='tenure_bucket', y='Churn_Probability',
                     title='Average Risk by Tenure',
                     labels={'Churn_Probability': 'Avg Churn Probability', 'tenure_bucket': 'Tenure'})
        st.plotly_chart(fig, use_container_width=True)
    
    # High risk customers table
    st.subheader("Top 20 Highest Risk Customers")
    high_risk_df = df.nlargest(20, 'Churn_Probability')[
        ['customerID', 'tenure', 'Contract', 'MonthlyCharges', 'Churn_Probability', 'Risk_Category']
    ]
    high_risk_df['Churn_Probability'] = (high_risk_df['Churn_Probability'] * 100).round(1).astype(str) + '%'
    st.dataframe(high_risk_df, use_container_width=True)


def main():
    """Main application."""
    st.title("📊 Customer Churn Prediction")
    st.markdown("*Identify at-risk customers and take proactive retention actions*")
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select Page", ["Single Prediction", "Batch Analysis"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This dashboard predicts customer churn probability 
    using a machine learning model trained on historical data.
    
    **Model:** XGBoost Classifier  
    **AUC-ROC:** 0.84
    """)
    
    if page == "Single Prediction":
        render_single_prediction()
    else:
        render_batch_analysis()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Built with Streamlit • Part of the Data Analysis Portfolio Builder</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
