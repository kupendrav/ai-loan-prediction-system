# Predict loan approval risk to save $151,000+ annually through faster screening, reduced manual review, and avoided high-risk approvals

## Executive Summary

This Flask-based loan risk screening system is powered by a Gaussian Naive Bayes machine learning model. It helps lending, credit, and risk teams make faster first-level loan decisions by estimating whether an applicant should be marked as accepted or rejected based on applicant profile, loan details, credit score, income, employment experience, and previous default history.

The solution is designed as a decision-support layer, not a replacement for final credit governance. It can reduce manual screening workload, improve consistency in loan evaluation, identify risky applications earlier, and help the organization protect revenue by reducing operational cost and avoidable credit exposure.

## Business Problem

Traditional loan screening often depends on manual checks, fixed rule-based filters, and delayed risk review. This creates several business challenges:

- High operational cost due to repeated manual review of similar applications.
- Slower loan processing, which can reduce customer conversion.
- Inconsistent decisions across teams, branches, or reviewers.
- Missed risk patterns when decisions rely only on static rules.
- Revenue leakage from approving weak applications or spending review time on clearly low-fit cases.

This model addresses the first decision layer by quickly classifying applications using historical loan data.

## Proposed Solution

The application provides a web interface where a credit officer or business user can enter borrower and loan information. The trained model then returns:

- Predicted loan status: `Accepted` or `Rejected`.
- Probability score behind the prediction.
- Key supporting and risk reasons.
- Model metrics from the saved training artifact.

The system uses a trained Gaussian Naive Bayes classifier with threshold tuning. The model artifact stores preprocessing configuration, feature columns, default values, category options, decision threshold, and evaluation metrics so the application can serve predictions consistently.

## Model Performance

The current model was trained and evaluated on the included `loan_data.csv` dataset.

| Metric | Value |
| --- | ---: |
| Original records | 45,000 |
| Records after outlier removal | 37,992 |
| Records removed as outliers | 7,008 |
| Validation accuracy | 87.88% |
| Test accuracy | 87.10% |
| Decision threshold | 0.61 |
| Model type | Gaussian Naive Bayes |

Confusion matrix on the test set:

| Actual / Predicted | Predicted 0 | Predicted 1 |
| --- | ---: | ---: |
| Actual 0 | 5,439 | 493 |
| Actual 1 | 487 | 1,180 |

Classification summary:

- Class `0` precision: 91.78%
- Class `0` recall: 91.69%
- Class `1` precision: 70.53%
- Class `1` recall: 70.79%
- Weighted F1-score: 87.11%

## Estimated Annual Revenue Saved: $151,000+

The system saves money and protects revenue in four main ways:

1. **Lower manual review cost**
   Applications can be pre-screened instantly, allowing credit teams to focus human review on borderline or high-value cases.

2. **Reduced turnaround time**
   Faster decisions improve customer experience and reduce drop-off during loan application processing.

3. **Better risk filtering**
   Risk indicators such as low credit score, high loan-to-income ratio, high interest rate, low employment experience, and previous default history are evaluated consistently.

4. **Improved process consistency**
   The same model and threshold are applied to every application, reducing decision variance across reviewers.

## Financial Impact Breakdown

Actual savings depend on loan volume, manual review cost, approval policy, default loss rate, and how the business deploys the model. The following estimate is provided as a practical business scenario.

### Manual Review Cost Saving

Assumptions:

- 100,000 loan applications per year.
- Manual review time reduced by 10 minutes per application.
- Reviewer cost is approximately $5.27 per hour.

Estimated annual operational saving:

```text
100,000 applications x 10 minutes = 1,000,000 minutes saved
1,000,000 / 60 = 16,666.67 hours saved
16,666.67 x $5.27 = $87,800 annual saving
```

Estimated operational saving: **approximately $87,800 per year**.

### Credit Loss Avoidance

Assumptions:

- Average loan amount: approximately $1,054.
- The model helps avoid 300 high-risk approvals per year.
- Estimated loss exposure per avoided approval: 20% of loan amount.

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

This is an illustrative estimate converted for international presentation. For production use, the organization should replace assumptions with actual application volume, employee cost, average loan size, default rate, recovery rate, and approval policy.

## Key Use Cases

- **Loan pre-screening:** Quickly classify incoming applications before manual review.
- **Credit officer decision support:** Provide a probability score and reason summary to support faster judgement.
- **Branch-level consistency:** Standardize first-level loan evaluation across teams and regions.
- **Risk queue prioritization:** Route high-risk or uncertain cases to senior underwriters.
- **Customer journey optimization:** Give faster responses for clear approval or rejection cases.
- **Portfolio quality improvement:** Reduce exposure to applicants with weak repayment indicators.
- **Training and analytics:** Help new analysts understand which features influence risk decisions.

## What Makes This Solution Special

- **Lightweight and fast:** Naive Bayes is computationally efficient and suitable for quick screening.
- **Easy to deploy:** The model is packaged into a Flask app with a simple browser-based interface.
- **Transparent business signals:** The app provides reason summaries using understandable risk factors.
- **Threshold tuning included:** The decision threshold is optimized during training instead of relying only on the default `0.50`.
- **Reusable model artifact:** The saved artifact includes model, threshold, feature schema, defaults, category options, and metrics.
- **Low infrastructure cost:** The current solution can run on a basic server or local machine.
- **Good baseline accuracy:** The current model reaches 87.10% test accuracy on the processed dataset.

## Comparison With Other Approaches

| Approach | Strength | Limitation | Advantage of This Solution |
| --- | --- | --- | --- |
| Manual review only | Human judgement and policy awareness | Slow, costly, inconsistent | Automates first-level screening |
| Rule-based system | Easy to explain | Rigid and hard to maintain as patterns change | Learns from historical data |
| Logistic regression | Interpretable and strong baseline | May require more feature engineering | Naive Bayes is simpler and faster for baseline deployment |
| Random forest / boosting | Often higher predictive power | More complex, heavier, less transparent | Lightweight and easier to explain |
| External credit scoring API | Mature third-party data | Cost, dependency, data-sharing concerns | Runs internally with owned data |

## Challenges Faced

- **Data quality:** The dataset contains outliers that could distort model learning.
- **Outlier treatment:** IQR-based filtering was added to remove extreme values from important numeric fields.
- **Categorical encoding:** Gender, education, home ownership, loan intent, and previous default history required encoding before training.
- **Threshold selection:** The default classification threshold may not produce the best business result, so threshold tuning was implemented.
- **Class performance difference:** Class `1` has lower precision and recall than class `0`, meaning further improvement is needed before high-stakes autonomous approval.
- **Explainability:** Naive Bayes provides probabilities, but business users still need readable reasons, so the app adds a practical explanation layer.
- **Production readiness:** The current implementation is suitable as a prototype or decision-support tool. A production rollout should include monitoring, audit logs, bias checks, security controls, and periodic retraining.

## Risk and Governance Considerations

For real organizational deployment, the model should be governed carefully:

- Do not use the prediction as the only approval authority for regulated lending decisions.
- Add human review for borderline cases and high-value loans.
- Monitor model drift as borrower behavior and market conditions change.
- Validate fairness across protected or sensitive customer groups.
- Maintain audit logs for input data, prediction output, probability, timestamp, and reviewer action.
- Retrain periodically using fresh approved production data.
- Align final decisions with legal, compliance, and credit policy requirements.

## Technical Overview

### Features Used

The model uses the following borrower and loan attributes:

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

## Retrain the Model

```powershell
python scripts\train_model.py
```

The training script reads `loan_data.csv`, encodes categorical variables, removes outliers, trains the Gaussian Naive Bayes model, tunes the decision threshold, evaluates performance, and saves the final artifact to:

```text
models/loan_status_new_model.pkl
```

## Recommended Next Steps

- Validate model performance on recent production data.
- Add ROC-AUC, precision-recall curve, and business-cost-based threshold selection.
- Compare against logistic regression, random forest, and gradient boosting models.
- Add database storage for prediction history and audit trails.
- Add authentication and role-based access for business users.
- Add monitoring for drift, approval rate changes, and performance degradation.
- Create a business dashboard for monthly savings, approval quality, and review time reduction.

## Conclusion

This loan status prediction system provides a practical, low-cost business solution for faster and more consistent loan screening. With 87.10% test accuracy on the processed dataset and a simple web interface, it can help organizations reduce manual workload, improve operational efficiency, and protect revenue from avoidable credit risk.

The current implementation is best positioned as a decision-support model. With additional validation, governance, monitoring, and integration into the loan origination workflow, it can become a valuable production-ready risk intelligence layer.
#   a i - l o a n - p r e d i c t i o n - s y s t e m  
 