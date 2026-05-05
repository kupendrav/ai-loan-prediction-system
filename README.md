# Predict Loan Approval Risk to Save $151,000+ Annually

A Flask-based loan risk screening system powered by a Gaussian Naive Bayes model. Designed as a **decision-support layer** to accelerate first-level screening, reduce manual review, and avoid high-risk approvals.

---

## Executive Summary

Traditional loan screening often relies on manual checks and static rules, leading to slow turnaround, inconsistent decisions, and preventable risk exposure. This application provides a lightweight ML-based pre-screening layer that:

- Classifies applications as **Accepted** or **Rejected**
- Returns a **probability score** behind the decision
- Shows **key supporting and risk reasons**
- Reports **saved model metrics** and configuration

> Governance note: This system supports credit decisions; it does **not** replace final underwriting or regulatory governance.

---

## Business Problem

Manual and rule-based screening commonly creates:

- High operational cost due to repetitive manual review
- Slower processing that reduces customer conversion
- Inconsistent decisions across reviewers or branches
- Missed risk patterns due to static rules
- Revenue leakage from weak approvals or wasted review time

This model improves the *first decision layer* by classifying applications quickly using historical loan data.

---

## Proposed Solution

A web interface allows credit officers or business users to enter borrower and loan details. The model then returns:

- Predicted loan status: `Accepted` or `Rejected`
- Probability score for the predicted class
- Key supporting / risk reasons (human-readable)
- Model metrics and metadata from the saved artifact

The system uses a trained **Gaussian Naive Bayes** classifier with **threshold tuning**. The model artifact stores:

- Preprocessing configuration
- Feature columns and defaults
- Category options
- Decision threshold
- Model metrics

---

## Model Performance

The current model was trained and evaluated using the included dataset: `loan_data.csv`.

| Metric | Value |
| --- | ---: |
| Original records | 45,000 |
| Records after outlier removal | 37,992 |
| Records removed as outliers | 7,008 |
| Validation accuracy | 87.88% |
| Test accuracy | 87.10% |
| Decision threshold | 0.61 |
| Model type | Gaussian Naive Bayes |

### Confusion Matrix (Test Set)

| Actual / Predicted | Predicted 0 | Predicted 1 |
| --- | ---: | ---: |
| Actual 0 | 5,439 | 493 |
| Actual 1 | 487 | 1,180 |

### Classification Summary

- Class `0` precision: 91.78%
- Class `0` recall: 91.69%
- Class `1` precision: 70.53%
- Class `1` recall: 70.79%
- Weighted F1-score: 87.11%

---

## Estimated Annual Revenue Saved: $151,000+

This system saves money and protects revenue in four main ways:

1. **Lower manual review cost**
   Applications can be pre-screened instantly, allowing reviewers to focus on borderline or high-value cases.

2. **Reduced turnaround time**
   Faster first-level decisions improve customer experience and reduce drop-off.

3. **Better risk filtering**
   Risk indicators (credit score, loan-to-income ratio, interest rate, employment experience, default history) are evaluated consistently.

4. **Improved process consistency**
   A consistent model and threshold reduce decision variance across reviewers.

---

## Financial Impact Breakdown

Actual savings depend on volume, review cost, approval policy, and default loss rate. The following estimate provides a practical business scenario.

### 1) Manual Review Cost Saving

Assumptions:

- 100,000 loan applications per year
- Manual review time reduced by 10 minutes per application
- Reviewer cost is approximately $5.27 per hour

Estimated annual operational saving:

```text
100,000 applications x 10 minutes = 1,000,000 minutes saved
1,000,000 / 60 = 16,666.67 hours saved
16,666.67 x $5.27 = $87,800 annual saving
```

Estimated operational saving: **approximately $87,800 per year**.

### 2) Credit Loss Avoidance

Assumptions:

- Average loan amount: approximately $1,054
- The model helps avoid 300 high-risk approvals per year
- Estimated loss exposure per avoided approval: 20% of loan amount

Estimated annual credit loss avoided:

```text
300 avoided risky approvals x $1,054 x 20% = $63,240
```

Estimated credit loss avoidance: **approximately $63,200 per year**.

### Combined Business Impact

```text
$87,800 operational saving + $63,200 credit loss avoidance
= $151,000 estimated annual revenue saved
```

This is an illustrative estimate. For production use, replace assumptions with actual loan volume, employee cost, average loan size, and observed default rates.

---

## Key Use Cases

- **Loan pre-screening:** Quickly classify incoming applications before manual review
- **Credit officer decision support:** Use probability score + reasons to support judgement
- **Branch-level consistency:** Standardize first-level evaluation across teams
- **Risk queue prioritization:** Route high-risk or uncertain cases to senior underwriters
- **Customer journey optimization:** Respond faster for clear approvals or rejections
- **Portfolio quality improvement:** Reduce exposure to weak repayment indicators
- **Training and analytics:** Help new analysts understand risk drivers

---

## What Makes This Solution Special

- **Lightweight and fast:** Naive Bayes is computationally efficient for quick screening
- **Easy to deploy:** Packaged as a Flask app with a browser-based interface
- **Transparent business signals:** Adds readable “reasons” on top of model probabilities
- **Threshold tuning included:** Optimizes the decision threshold beyond the default `0.50`
- **Reusable model artifact:** Stores model + schema + defaults + threshold + metrics
- **Low infrastructure cost:** Runs on a basic server or local machine
- **Good baseline accuracy:** 87.10% test accuracy on processed dataset

---

## Comparison With Other Approaches

| Approach | Strength | Limitation | Advantage of This Solution |
| --- | --- | --- | --- |
| Manual review only | Human judgement and policy awareness | Slow, costly, inconsistent | Automates first-level screening |
| Rule-based system | Easy to explain | Rigid and hard to maintain | Learns from historical data |
| Logistic regression | Interpretable baseline | Needs more feature engineering | Simpler baseline deployment |
| Random forest / boosting | Often higher accuracy | Heavier, less transparent | Lightweight and explainable |
| External credit scoring API | Mature third-party data | Cost & dependency risks | Runs internally on owned data |

---

## Challenges Faced

- **Data quality:** Outliers can distort learning
- **Outlier treatment:** IQR filtering removes extremes from numeric fields
- **Categorical encoding:** Required for gender, education, home ownership, intent, defaults
- **Threshold selection:** Implemented threshold tuning for business outcomes
- **Class imbalance/performance gap:** Class `1` underperforms class `0`
- **Explainability:** Converts model probabilities into readable business reasons
- **Production readiness:** Prototype-quality; production needs monitoring, audit logs, bias checks, and security controls

---

## Risk and Governance Considerations

For real deployment, the model should be governed carefully:

- Do not use prediction as the sole approval authority
- Add human review for borderline / high-value loans
- Monitor model drift over time
- Validate fairness across sensitive/protected groups
- Maintain audit logs (inputs, outputs, probability, timestamp, reviewer action)
- Retrain periodically using fresh production data
- Align decisions with legal, compliance, and credit policy

---

## Technical Overview

### Features Used

- Age
- Gender
- Education
- Annual income
- Employment experience
- Home ownership
- Loan amount
- Loan intent
- Loan interest rate
- Loan percent income
- Credit history length
- Credit score
- Previous loan defaults on file

### Application Structure

```text
.
|-- app.py
|-- loan_data.csv
|-- models/
|   `-- loan_status_new_model.pkl
|-- scripts/
|   `-- train_model.py
|-- static/
|   `-- css/
|-- templates/
|   `-- index.html
|-- test_app.py
|-- requirements.txt
`-- README.md
```

---

## Run the Application

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run locally:

```powershell
python app.py
```

Open the application:

```text
http://127.0.0.1:5000/
```

For production-style serving, use the WSGI app with Waitress:

```powershell
waitress-serve --host 0.0.0.0 --port 5000 app:app
```

Run the included app checks:

```powershell
python test_app.py
```

---

## Retrain the Model

```powershell
python scripts\train_model.py
```

The training script reads `loan_data.csv`, encodes categorical variables, removes outliers, trains the Gaussian Naive Bayes model, tunes the decision threshold, evaluates performance, and saves:

```text
models/loan_status_new_model.pkl
```

---

## Recommended Next Steps

- Validate performance on recent production data
- Add ROC-AUC and precision-recall curves
- Use business-cost-based threshold selection
- Compare with logistic regression, random forest, gradient boosting
- Add database storage for prediction history + audit trails
- Add authentication and role-based access
- Add monitoring for drift, approval rate changes, and performance degradation
- Add a business dashboard for savings and review-time reduction

---

## Conclusion

This loan status prediction system is a practical, low-cost decision-support solution for faster and more consistent first-level screening. With **87.10% test accuracy** and a simple web interface, it can reduce operational workload and help avoid risky approvals.

For production rollout, apply governance controls, monitoring, fairness validation, and integration into the loan origination workflow.
