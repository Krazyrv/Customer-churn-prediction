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
    pass

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    pass

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