# Customer Churn Prediction

## Case Study

---

### TL;DR

Built a machine learning model to predict customer churn with 84% AUC-ROC, enabling the retention team to identify 49% of churners by targeting just the top 20% highest-risk customers—a 2.5x improvement over random outreach.

---

### Role & Timeline

**Role:** Data Scientist (Solo Project)
**Timeline:** 4 weeks
**Responsibilities:** EDA, feature engineering, model selection, hyperparameter tuning, deployment, business recommendations

---

### Business Context

A telecommunications company was experiencing 26% annual customer churn, costing approximately $1.5M in lost revenue. The retention team ran campaigns by randomly selecting customers, resulting in:

- Wasted budget on customers who weren't going to leave
- Missing high-risk customers who churned before intervention
- No way to prioritize limited retention resources

**Goal:** Build a predictive model to identify at-risk customers so the retention team can intervene proactively and prioritize their efforts.

---

### Data & Methods

**Dataset:** IBM Telco Customer Churn (public dataset)

- 7,043 customers with 21 features
- 26.5% churn rate (imbalanced)
- Mix of demographics, account info, and services

**Feature Engineering:**

- Tenure buckets (0-12, 12-24, 24-48, 48+ months)
- Service count aggregations
- Contract and payment risk scores
- Charges-to-tenure ratios

**Modeling:**

- Compared Logistic Regression, Random Forest, XGBoost, LightGBM
- Used 5-fold stratified cross-validation
- Optimized for AUC-ROC due to class imbalance
- Selected XGBoost for best performance

**Stack:** Python (Scikit-learn, XGBoost, SHAP, Streamlit)

---

### Results

| Metric                | Value | Business Meaning                  |
| --------------------- | ----- | --------------------------------- |
| **AUC-ROC**     | 0.84  | Strong ranking ability            |
| **Lift @20%**   | 2.5x  | 2.5x more efficient than random   |
| **Recall @20%** | 49%   | Catch half of churners in top 20% |

**Top 3 Churn Drivers:**

1. **Month-to-month contract** — 3x higher churn than annual contracts
2. **Tenure < 12 months** — New customers at highest risk
3. **No tech support** — 2x higher churn for internet customers

![Model Performance](img/model_performance.png)

---

### Recommendations

1. **Target month-to-month customers** with contract upgrade offers (20% discount for annual)
2. **Implement 30/60/90 day onboarding** check-ins for new customers
3. **Bundle tech support** as free add-on for first 6 months
4. **Investigate fiber optic service** issues (higher churn than DSL)

**Expected Impact:** 15-20% reduction in churn, saving $225K-$300K annually

---

### Technical Highlights

- Handled class imbalance with stratified sampling
- Used SHAP for model explainability
- Built interactive Streamlit dashboard for real-time predictions
- Threshold optimization for business cost constraints

---

### Next Steps

With more time, I would: 

1. Build survival model to predict *when* churn occurs
2. Add calibration for more reliable probability estimates
3. Design A/B test to measure retention campaign impact
4. Create automated retraining pipeline with drift detection

---

**Code:** [Github](https://github.com/Krazyrv/Customer-churn-prediction)
**Demo:** [Dashboard link]()
**Contact:** [anthonynguyen1422@gmail.com](anthonynguyen1422@gmail.com)
