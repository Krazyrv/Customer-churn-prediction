"""
Generate Sample Telco Customer Churn Data
Creates a realistic sample dataset for testing without downloading from Kaggle.
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

N_CUSTOMERS = 7043  # Same as original dataset

def generate_sample_data():
    """Generate synthetic Telco churn data."""
    print("🔄 Generating sample Telco customer data...")
    
    # Customer IDs
    customer_ids = [f'{i:04d}-{"".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 5))}' 
                    for i in range(N_CUSTOMERS)]
    
    # Demographics
    gender = np.random.choice(['Male', 'Female'], N_CUSTOMERS)
    senior_citizen = np.random.choice([0, 1], N_CUSTOMERS, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], N_CUSTOMERS, p=[0.48, 0.52])
    dependents = np.random.choice(['Yes', 'No'], N_CUSTOMERS, p=[0.30, 0.70])
    
    # Tenure (in months, 0-72)
    # New customers more likely, long tenure less likely
    tenure = np.random.exponential(scale=25, size=N_CUSTOMERS).astype(int)
    tenure = np.clip(tenure, 0, 72)
    
    # Phone service
    phone_service = np.random.choice(['Yes', 'No'], N_CUSTOMERS, p=[0.90, 0.10])
    
    # Multiple lines (only if phone service)
    multiple_lines = np.where(
        phone_service == 'No',
        'No phone service',
        np.random.choice(['Yes', 'No'], N_CUSTOMERS, p=[0.42, 0.58])
    )
    
    # Internet service
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], N_CUSTOMERS, p=[0.34, 0.44, 0.22])
    
    # Internet-dependent services
    def internet_dependent(internet, yes_prob=0.35):
        if internet == 'No':
            return 'No internet service'
        return np.random.choice(['Yes', 'No'], p=[yes_prob, 1-yes_prob])
    
    online_security = [internet_dependent(i, 0.29) for i in internet_service]
    online_backup = [internet_dependent(i, 0.34) for i in internet_service]
    device_protection = [internet_dependent(i, 0.34) for i in internet_service]
    tech_support = [internet_dependent(i, 0.29) for i in internet_service]
    streaming_tv = [internet_dependent(i, 0.38) for i in internet_service]
    streaming_movies = [internet_dependent(i, 0.39) for i in internet_service]
    
    # Contract type
    contract = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'], 
        N_CUSTOMERS, 
        p=[0.55, 0.21, 0.24]
    )
    
    # Paperless billing
    paperless_billing = np.random.choice(['Yes', 'No'], N_CUSTOMERS, p=[0.59, 0.41])
    
    # Payment method
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        N_CUSTOMERS,
        p=[0.34, 0.23, 0.22, 0.21]
    )
    
    # Monthly charges (depends on services)
    base_charge = 20
    monthly_charges = np.zeros(N_CUSTOMERS)
    
    for i in range(N_CUSTOMERS):
        charge = base_charge
        if phone_service[i] == 'Yes':
            charge += np.random.uniform(15, 25)
        if multiple_lines[i] == 'Yes':
            charge += np.random.uniform(5, 15)
        if internet_service[i] == 'DSL':
            charge += np.random.uniform(20, 35)
        elif internet_service[i] == 'Fiber optic':
            charge += np.random.uniform(35, 55)
        if online_security[i] == 'Yes':
            charge += np.random.uniform(5, 10)
        if online_backup[i] == 'Yes':
            charge += np.random.uniform(5, 10)
        if device_protection[i] == 'Yes':
            charge += np.random.uniform(5, 10)
        if tech_support[i] == 'Yes':
            charge += np.random.uniform(5, 10)
        if streaming_tv[i] == 'Yes':
            charge += np.random.uniform(10, 15)
        if streaming_movies[i] == 'Yes':
            charge += np.random.uniform(10, 15)
        
        monthly_charges[i] = round(charge + np.random.uniform(-5, 5), 2)
    
    # Total charges
    total_charges = tenure * monthly_charges
    # Some variation
    total_charges = total_charges * np.random.uniform(0.95, 1.05, N_CUSTOMERS)
    total_charges = np.round(total_charges, 2)
    # New customers have blank total charges sometimes
    total_charges = np.where(tenure == 0, '', total_charges.astype(str))
    
    # Churn - based on risk factors (target ~26% churn rate)
    churn_prob = np.zeros(N_CUSTOMERS)
    
    for i in range(N_CUSTOMERS):
        prob = 0.08  # lower base rate
        
        # Contract is biggest factor
        if contract[i] == 'Month-to-month':
            prob += 0.18
        elif contract[i] == 'One year':
            prob += 0.03
        
        # Tenure - new customers churn more
        if tenure[i] < 12:
            prob += 0.10
        elif tenure[i] > 48:
            prob -= 0.05
        
        # Fiber optic churns more
        if internet_service[i] == 'Fiber optic':
            prob += 0.06
        
        # No tech support
        if tech_support[i] == 'No' and internet_service[i] != 'No':
            prob += 0.04
        
        # Electronic check
        if payment_method[i] == 'Electronic check':
            prob += 0.06
        
        # Senior citizens churn slightly more
        if senior_citizen[i] == 1:
            prob += 0.03
        
        # High charges
        if monthly_charges[i] > 80:
            prob += 0.03
        
        churn_prob[i] = np.clip(prob, 0, 0.85)
    
    churn = np.random.binomial(1, churn_prob).astype(str)
    churn = np.where(churn == '1', 'Yes', 'No')
    
    # Create DataFrame
    df = pd.DataFrame({
        'customerID': customer_ids,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn
    })
    
    print(f"✅ Generated {len(df):,} customers")
    print(f"   Churn rate: {(df['Churn'] == 'Yes').mean()*100:.1f}%")
    
    return df


def main():
    """Generate and save sample data."""
    df = generate_sample_data()
    
    # Save
    project_root = Path(__file__).parent.parent.parent
    
    # Save to raw
    raw_dir = project_root / 'data' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_dir / 'WA_Fn-UseC_-Telco-Customer-Churn.csv', index=False)
    print(f"💾 Saved to: {raw_dir / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'}")
    
    # Save to sample
    sample_dir = project_root / 'data' / 'sample'
    sample_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(sample_dir / 'sample_data.csv', index=False)
    print(f"💾 Saved to: {sample_dir / 'sample_data.csv'}")


if __name__ == '__main__':
    main()
