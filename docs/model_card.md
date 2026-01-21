# Model Card: Customer Churn Prediction

## Model Details

**Model Name:** Telco Customer Churn Predictor
**Version:** 1.0
**Type:** Binary Classification
**Algorithm:** XGBoost / Gradient Boosting Classifier
**Framework:** Scikit-learn, XGBoost
**Date:** January 2026

---

## Intended Use

### Primary Use Case

Predict the probability that a telecommunications customer will cancel their subscription (churn) within the next billing cycle.

### Intended Users

- Customer Retention Team
- Marketing Department
- Customer Success Managers

### Out-of-Scope Uses

- Decisions about customer pricing or service terms
- Automated contract termination
- Any use without human review of predictions

---

## Training Data

**Source:** IBM Telco Customer Churn Dataset
**Size:** 7,043 customers
**Time Period:** Historical snapshot (not longitudinal)
**Features:** 21 original, 54 after encoding

### Feature Categories

- **Demographics:** Gender, Senior Citizen, Partner, Dependents
- **Account:** Tenure, Contract Type, Payment Method, Billing
- **Services:** Phone, Internet, Security, Streaming, Support
- **Charges:** Monthly and Total charges

### Target Variable

- **Churn:** Binary (Yes/No)
- **Prevalence:** 26.5% positive class

---

## Evaluation Metrics

### Overall Performance

| Metric            | Value |
| ----------------- | ----- |
| AUC-ROC           | 0.84  |
| Average Precision | 0.71  |
| Accuracy          | 0.78  |

### At Default Threshold (0.5)

| Metric    | Value |
| --------- | ----- |
| Precision | 0.63  |
| Recall    | 0.47  |
| F1 Score  | 0.54  |

### Business Metrics

| Metric      | Value |
| ----------- | ----- |
| Lift @10%   | 2.8x  |
| Lift @20%   | 2.5x  |
| Recall @20% | 49%   |

---

## Ethical Considerations

### Potential Biases

- **Senior Citizens:** Model shows higher churn prediction for seniors. Ensure retention offers don't discriminate.
- **Gender:** Model includes gender but shows minimal impact. Monitor for disparate treatment.
- **Payment Method:** Electronic check users flagged as higher risk. May correlate with income.

### Fairness Analysis

The model was analyzed for disparate impact across protected groups. No significant differences in false positive rates were found, but continuous monitoring is recommended.

### Mitigation Strategies

1. Do not use predictions for pricing or service denial
2. Human review required before any customer contact
3. Regular fairness audits on new data
4. Transparent explanation to customers if asked

---

## Limitations

### Known Limitations

1. **Snapshot Data:** Trained on static data, doesn't capture temporal patterns
2. **Single Company:** May not generalize to other telecoms or industries
3. **Feature Drift:** Customer behavior may change over time
4. **No Causal Claims:** Correlations only, not causal relationships

### When Not to Use

- Real-time, high-frequency predictions without monitoring
- Fully automated decision-making without human oversight
- Markets or customer segments not represented in training data

---

## Monitoring

### Recommended Monitoring

1. **Prediction drift:** Monitor distribution of predicted probabilities weekly
2. **Feature drift:** Track input feature distributions monthly
3. **Performance decay:** Recalculate AUC-ROC on labeled data quarterly
4. **Fairness:** Audit across demographic groups quarterly

### Retraining Triggers

- AUC-ROC drops below 0.75
- Significant feature distribution shift detected
- Major business changes (new products, pricing)
- Every 6 months at minimum

---

## Model Explainability

### Global Explanations

Top features by importance:

1. Contract type (Month-to-month)
2. Tenure (months as customer)
3. Monthly charges
4. Internet service type
5. Tech support subscription

### Local Explanations

SHAP values available for individual predictions, showing which features increased or decreased churn probability for each customer.

---

## Technical Specifications

### Input Format

```python
{
    'tenure': int,  # 0-72
    'MonthlyCharges': float,  # 0-200
    'Contract': str,  # 'Month-to-month', 'One year', 'Two year'
    # ... other features
}
```

### Output Format

```python
{
    'churn_probability': float,  # 0.0-1.0
    'risk_category': str,  # 'Low', 'Medium', 'High'
    'recommendation': str  # Action to take
}
```

### Performance Requirements

- Inference time: <100ms per prediction
- Memory: <500MB model size
- Dependencies: scikit-learn, xgboost, pandas, numpy

---

## Contact

**Model Owner:** Anthony Nguyen
**Email:** anthonynguyen14202@gmail.com
**Last Updated:** January 2026

---

## Version History

| Version | Date     | Changes         |
| ------- | -------- | --------------- |
| 1.0     | Jan 2026 | Initial release |
