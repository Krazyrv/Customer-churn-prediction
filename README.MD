# Customer Churn Prediction

**Predict which customers will cancel their subscription and enable proactive retention campaigns that reduce churn by identifying at-risk customers before they leave.**

![Model Performance](src/docs/img/model_performance.png)

---

## TL;DR

- **What:** Machine learning model predicting customer churn with 80%+ accuracy
- **Business Impact:** Identify 92% of churners in the top 40% risk scores, enabling targeted retention with 2.4x efficiency vs. random outreach
- **Stack:** Python, Scikit-learn, XGBoost, SHAP, Streamlit

---

## Quickstart

```bash
# Clone and setup
git clone <your-repo-url>
cd Customer-churn-prediction 
pip install -r requirements.txt

# Train model and run predictions
python src/pipelines/train.py

# Launch the prediction dashboard
streamlit run demo/app.py
```

**Time to run:** Under 5 minutes

---

## Business Problem

A telecommunications company loses **$1.5M annually** to customer churn. The marketing team currently runs retention campaigns by randomly selecting customers, resulting in wasted spend on customers who weren't going to leave anyway.

**The Challenge:**

- 26.5% of customers churn (high baseline)
- Retention campaigns cost $50/customer
- Acquiring a new customer costs 5x more than retaining an existing one
- No way to prioritize which customers to target

**The Goal:**
Build a model that identifies high-risk customers so the retention team can:

1. Focus limited budget on customers most likely to leave
2. Intervene *before* the customer decides to cancel
3. Measure and improve retention campaign ROI

---

## Dataset

**Source:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Feature          | Description                    |
| ---------------- | ------------------------------ |
| customerID       | Unique customer identifier     |
| gender           | Male/Female                    |
| SeniorCitizen    | Whether customer is 65+        |
| Partner          | Has partner (Yes/No)           |
| Dependents       | Has dependents (Yes/No)        |
| tenure           | Months as customer             |
| PhoneService     | Has phone service              |
| MultipleLines    | Has multiple lines             |
| InternetService  | DSL, Fiber optic, or No        |
| OnlineSecurity   | Has online security add-on     |
| OnlineBackup     | Has online backup add-on       |
| DeviceProtection | Has device protection          |
| TechSupport      | Has tech support add-on        |
| StreamingTV      | Has streaming TV               |
| StreamingMovies  | Has streaming movies           |
| Contract         | Month-to-month, 1 year, 2 year |
| PaperlessBilling | Uses paperless billing         |
| PaymentMethod    | Payment method used            |
| MonthlyCharges   | Monthly charge amount          |
| TotalCharges     | Total amount charged           |
| Churn            | Target: Yes/No                 |

**Size:** 7,043 customers | 21 features | 26.5% churn rate

---

## Key Results

### 📊KEY FINDINGS SUMMARY

##### 1. OVERALL CHURN

   • Churn Rate: 26.5%
   • Total Customers at Risk: 1,869

##### 2. TOP CHURN DRIVERS (Actionable)

   a) CONTRACT TYPE
   • Month-to-month: ~43% churn
   • Action: Offer discounts for annual contracts

   b) TENURE
   • New customers (0-12 mo): ~48% churn
   • Action: Onboarding program, check-ins at 30/60/90 days

   c) TECH SUPPORT
   • Without tech support: 2x higher churn
   • Action: Bundle free tech support for first 6 months

   d) INTERNET SERVICE
   • Fiber optic: ~42% churn
   • Action: Investigate service quality, set expectations

##### 3. LOW-IMPACT FEATURES

   • Gender: No significant difference
   • Phone service: Minimal impact

##### 4. MODELING CONSIDERATIONS

   • Class imbalance: Use stratified sampling, SMOTE
   • Key features: Contract, Tenure, Services, Charges
   • Evaluation: Use AUC-ROC, Precision-Recall (not just accuracy)

#### Target Variable Analysis

![Churn Distribution](docs/img/churn_distribution.png)

💡 INSIGHT: Class imbalance present - 26.5% churn rate
   Will need stratified sampling and appropriate evaluation metrics.

#### Feature Analysis

##### Demographics

![demographics churns](docs/img/demographics_churn.png)

💡 INSIGHTS:

- Gender has minimal impact on churn
- Senior citizens churn ~40% more
- Customers with partners/dependents churn less (more stable)

![contract churns](docs/img/contract_churn.png)

💡 KEY INSIGHT: Month-to-month customers churn 15.1x more than 2-year contracts!

- Month-to-month: 42.7% churn
- Two year: 2.8% churn

![tenure churn](docs/img/tenure_churn.png)

💡 INSIGHT: New customers (0-12 months) have 47.7% churn rate
   Long-term customers (48+ months) only 9.5% churn rate
   → Early engagement is critical for retention!

![internet_churn](docs/img/internet_churn.png)

⚠️ INSIGHT: Fiber optic customers churn at 41.9%!
   Possible reasons:

- Higher expectations not met?
- More tech-savvy, easier to switch?
- Service quality issues?

![support service churns](docs/img/support_services_churn.png)

💡 INSIGHT: Customers WITHOUT support services churn 2x more!
   → Bundle support services as retention strategy

![streaming services churns](docs/img/streaming_services_churn.png)

💡 INSIGHT: Streaming services result in slightly reduced churn.
   Possible reasons:

- High competition in streaming market
- Customers may prioritize content over service loyalty
  → Keep improving content and user experience
  → Consider exclusive content partnerships

![monthly charges churns](docs/img/charges_churn.png)

💡 INSIGHT: Churned customers pay more on average

- Churned: $74.44/month
- Retained: $61.27/month
  → Higher charges may lead to value perception issues

#### Correlation Analysis

![feature correlation](docs/img/correlations.png)

💡 KEY CORRELATIONS:
   🔴 INCREASES churn: Paperless Billing, Senior Citizen, Monthly Charges
   🟢 DECREASES churn: Contract length, Tenure, Partner, Dependents

### Model Performance

| Metric                   | Value | Business Meaning                         |
| ------------------------ | ----- | ---------------------------------------- |
| **AUC-ROC**        | 0.84  | Strong ranking ability                   |
| **Precision @20%** | 0.63  | 63% of top-risk customers actually churn |
| **Recall @20%**    | 0.47  | Captures 47% of all churners in top 20%  |
| **Lift @20%**      | 2.4x  | 2.4x better than random targeting        |

### Confusion Matrix (Threshold: 0.5)

```
              Predicted
              No    Yes
Actual No    1165   144
       Yes    226   274
```

### Top Churn Drivers (SHAP Analysis)

![Feature Importance](src/docs/img/feature_importance.png)

1. **Contract Type** — Month-to-month customers churn 3x more than annual contracts
2. **Tenure** — Customers < 12 months are highest risk
3. **Monthly Charges** — Higher charges correlate with higher churn
4. **Internet Service** — Fiber optic customers churn more (service issues?)
5. **Tech Support** — Customers without tech support are 2x more likely to churn

---

## Project Structure

```
02-customer-churn-prediction/
├── data/
│   ├── 00-sample		   # Sample for quick testing
│   ├── 01-raw/                    # Original dataset
│   └── 02-preprocessed/           # Cleaned, feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb      # Model training & evaluation
├── src/
│   └──	pipelines         		# Pipelines
│   	├── preprocess_pipeline.py       # Data cleaning & feature engineering
│   	├── train_pipeline.py              # Model training script
│   	├── predict.py            # Prediction utilities
│   	└── evaluate.py           # Model evaluation metrics
├── models/
│   ├── feature_names.pkl
│   └── churn_model.pkl       # Trained model artifact
├── demo/
│   └── app.py                # Streamlit prediction app
├── docs/
│   ├── case_study.md         # One-page case study
│   ├── model_card.md         # Model documentation
│   └── img/                  # Visualizations
├── requirements.txt
├── Makefile
└── README.md
```

---

## Approach

### 1. Exploratory Data Analysis

- Analyzed churn distribution across all features
- Identified class imbalance (26.5% churn)
- Found key correlations: contract type, tenure, charges

### 2. Feature Engineering

- Created tenure buckets (0-12, 12-24, 24-48, 48+ months)
- Calculated charges per month of tenure
- Encoded categorical variables (one-hot and target encoding)
- Created service count features

### 3. Model Selection

Compared multiple algorithms:

| Model               | AUC-ROC        | Training Time |
| ------------------- | -------------- | ------------- |
| Logistic Regression | 0.81           | 0.2s          |
| Random Forest       | 0.82           | 2.1s          |
| **XGBoost**   | **0.84** | 1.3s          |
| LightGBM            | 0.83           | 0.8s          |

Selected **XGBoost** for best balance of performance and interpretability.

### 4. Hyperparameter Tuning

- Used Optuna for Bayesian optimization
- 5-fold stratified cross-validation
- Optimized for AUC-ROC

### 5. Threshold Optimization

- Default threshold (0.5) optimized for accuracy
- Business threshold (0.35) optimized for recall to catch more churners
- Provided threshold analysis for different cost scenarios

---

## Model Explainability

### Global Feature Importance (SHAP)

```
Contract_Month-to-month    ████████████████████  0.42
tenure                     ███████████████       0.31
MonthlyCharges            ████████████          0.25
InternetService_Fiber     ███████████           0.23
TechSupport_No            ██████████            0.21
OnlineSecurity_No         █████████             0.19
PaymentMethod_Electronic  ████████              0.17
PaperlessBilling          ███████               0.14
```

### Local Explanations

For any individual customer, the model provides:

- Churn probability (0-100%)
- Top 5 factors increasing/decreasing risk
- Recommended retention actions

---

## Business Recommendations

### 1. Target Month-to-Month Customers

- Offer discounts for switching to annual contracts
- Expected impact: 15-20% reduction in churn for this segment

### 2. Onboard New Customers Proactively

- Customers < 12 months tenure are highest risk
- Implement 30/60/90 day check-in calls
- Expected impact: Improve first-year retention by 10%

### 3. Bundle Tech Support

- Customers without tech support churn 2x more
- Offer free tech support for first 6 months
- Expected impact: 8% churn reduction

### 4. Investigate Fiber Optic Issues

- Fiber customers churn more than DSL
- Likely service quality or expectation mismatch
- Recommendation: Customer satisfaction survey

---

## Reproduce the Analysis

```bash

Step by step:
python src/preprocessing.py     # Clean and engineer features
python src/train.py             # Train model
python src/evaluate.py          # Generate evaluation metrics
streamlit run demo/app.py       # Launch prediction dashboard
```

---

## Customization Ideas

1. **Different Algorithm:** Try neural networks or ensemble stacking
2. **Cost-Sensitive Learning:** Weight misclassification costs differently
3. **Time-Based Validation:** Use temporal splits instead of random
4. **Survival Analysis:** Predict *when* customers will churn, not just *if*
5. **A/B Test Framework:** Add simulation for retention campaign ROI

---

## Model Card

See [docs/model_card.md](docs/model_card.md) for:

- Intended use cases
- Limitations and biases
- Ethical considerations
- Performance across subgroups

---

## Skills Demonstrated

- **Classification modeling** with imbalanced data
- **Feature engineering** for tabular data
- **Model evaluation** beyond accuracy (precision, recall, lift)
- **Hyperparameter tuning** with cross-validation
- **Model explainability** with SHAP
- **Production packaging** with serialized models
- **Interactive demos** with Streamlit

---

## Next Steps

With more time, I would:

1. Build a **survival model** to predict *when* churn happens
2. Create an **automated retraining pipeline** with drift detection
3. Design an **A/B test** to measure retention campaign impact
4. Add **calibration** for more reliable probability estimates

---

## Author

Anthony Nguyen | [Toan Nguyen | LinkedIn](https://www.linkedin.com/in/toan-nguyen-5489952b8/) | [Krazyrv | Github](https://github.com/Krazyrv)
