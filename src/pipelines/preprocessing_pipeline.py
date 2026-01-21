"""
Preprocessing Module
This module contains functions for preprocessing the customer churn dataset.
Including Data Cleaning, Encoding Categorical Variables, and Feature Engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Load raw data from Telco churn CSV file."""
    if file_path is None:
        project_root = Path(__file__).parent.parent.parent
        file_path = project_root / 'data' / '01-raw' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

    print(f"📂 Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Fix TotalCharges (some are NA due to new customers)
    - Convert target variable 'Churn' to binary
    - Handle missing types
    """
    print("🧹 Cleaning data...")

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_total_charges = df['TotalCharges'].isna().sum()
    if missing_total_charges > 0:
        print(f"   Found {missing_total_charges} missing TotalCharges, filling with MonthlyCharges * tenure")
        df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'], inplace=True)

    # Convert 'Churn' to binary
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0).astype(int)

    # Convert 'SeniorCitizen' to Yes/No for consistency
    df['SeniorCitizen'] = df['SeniorCitizen'].map({1: 'Yes', 0: 'No'})

    print(f"   Churn rate: {df['Churn'].mean()*100:.1f}%")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features for modelling:
    - Tenure buckets
    - Charges per tenure
    - Service count
    """
    print("🔧 Engineering features...")

    # Tenure buckets
    df['tenure_bucket'] = pd.cut(df['tenure'], 
                                 bins=[0, 12, 24, 48, 60, 72],
                                 labels=['0-12', '12-24', '24-48', '48+'], 
                                 right=False)
    
    # Average monthly charges (for customers with tenure > 0)
    df['avg_monthly_charges'] = np.where(
        df['tenure'] > 0,
        df['TotalCharges'] / df['tenure'],
        df['MonthlyCharges']
    )

    # Charges to tenure ratio with small constant to avoid division by zero (Laplace smoothing)
    # Higher = Paying more
    df['charges_tenure_ratio'] = df['MonthlyCharges'] / (df['avg_monthly_charges'] + 1)

    # Count of services subscribed
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]

    def count_services(row):
        count = 0
        for col in service_cols:
            val = row[col]
            if val not in ['No', 'No internet service', 'No phone service']:
                count += 1
        return count
    
    df['num_services'] = df.apply(count_services, axis=1)

    # Has streaming services
    # df['has_streaming'] = df['StreamingTV'].isin(['Yes', 'No internet service']) | df['StreamingMovies'].isin(['Yes', 'No internet service'])

    df['has_streaming'] = ((df['StreamingTV'] == 'Yes') | 
                           (df['StreamingMovies'] == 'Yes')).astype(int)
    
    # Has any support/security services
    df['has_security_support'] = ((df['OnlineSecurity'] == 'Yes') | 
                                (df['OnlineBackup'] == 'Yes') | 
                                (df['TechSupport'] == 'Yes')).astype(int)


    # Contract risk score (Short-term contracts are riskier)
    df['contract_risk'] = df['Contract'].map({
        'Month-to-month': 3,
        'One year': 2,
        'Two year': 1
    })

    # Payment method risk score (Electronic checks are riskier)
    payment_risk = { 
        'Electronic check': 3,
        'Mailed check': 2,
        'Bank transfer (automatic)': 1,
        'Credit card (automatic)': 1
    }
    df['payment_risk'] = df['PaymentMethod'].map(payment_risk)


    # New customer flag (tenure <= 6 months)
    df['is_new_customer'] = (df['tenure'] <= 6).astype(int)

    print(f"   Created {8} new features.")

    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features for modeling:
    - Label Encoding for binary categories
    - One-Hot Encoding for multi-class categories
    Returns:
    - X: Feature matrix
    - y: Target vector
    - feature_names: List of feature names after encoding
    - Encoders
    """
    print("🔤 Encoding features...")

    # Column to drop (ID, target, derived features)
    drop_cols = ['customerID', 'Churn', 'tenure_bucket']
    
    # Get feature columns
    feature_df = df.drop(columns=drop_cols, errors='ignore')
    
    # Separate target variable
    y = df['Churn'].values

    # Identify categorical and numeric columns
    categorical_cols = feature_df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = feature_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # numeric_cols = feature_df.select_dtypes(include=['number']).columns.tolist()
    print(f"   Found {len(categorical_cols)} categorical and {len(numeric_cols)} numeric features.")

    # One-Hot Encode categorical variables
    feature_df = pd.get_dummies(feature_df, columns=categorical_cols, drop_first=True)

    # Get feature names
    



def prepare_data(raw_path: str = None, save: bool = True):
    """Prepare data for modeling."""
    
    df = load_raw_data(raw_path)

    df = clean_data(df)
    df = engineer_features(df)
    df = encode_features(df)

    return [],[],[],[],[]

def main():
    """Run preprocessing pipeline."""
    print("=" * 60)
    print("CUSTOMER CHURN - DATA PREPROCESSING")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test, feature_names = prepare_data()
    
    print("\n✅ Preprocessing complete!")
    print(f"   Features: {len(feature_names)}")
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")


if __name__ == '__main__':
    main()