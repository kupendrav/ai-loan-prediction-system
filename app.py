import pickle
import os
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "loan_status_new_model.pkl"

app = Flask(__name__)


LABELS = {
    "person_age": "Age",
    "person_gender": "Gender",
    "person_education": "Education",
    "person_income": "Annual Income (USD)",
    "person_emp_exp": "Employment Experience (Years)",
    "person_home_ownership": "Home Ownership",
    "loan_amnt": "Loan Amount (USD)",
    "loan_intent": "Loan Intent",
    "loan_int_rate": "Loan Interest Rate (%)",
    "loan_percent_income": "Loan as Percent of Income (%)",
    "cb_person_cred_hist_length": "Credit History Length (Years)",
    "credit_score": "Credit Score",
}

HELP_TEXT = {
    "person_income": "Example: 60000 means annual income of $60,000.",
    "loan_amnt": "Example: 10000 means a $10,000 loan request.",
    "loan_int_rate": "Enter the annual interest rate as a percent, for example 10.5.",
    "loan_percent_income": "Enter a percent, for example 5 means the loan is 5% of income.",
    "previous_loan_defaults_on_file": "Choose Yes only if the applicant has a previous loan default on file.",
}

OPTION_LABELS = {
    "person_gender": {
        "female": "Female",
        "male": "Male",
    },
    "person_home_ownership": {
        "MORTGAGE": "Mortgage",
        "OTHER": "Other",
        "OWN": "Own",
        "RENT": "Rent",
    },
    "loan_intent": {
        "DEBTCONSOLIDATION": "Debt Consolidation",
        "EDUCATION": "Education",
        "HOMEIMPROVEMENT": "Home Improvement",
        "MEDICAL": "Medical",
        "PERSONAL": "Personal",
        "VENTURE": "Business Venture",
    },
    "previous_loan_defaults_on_file": {
        "No": "No previous default",
        "Yes": "Has previous default",
    },
}

STEPS = {
    "person_age": "1",
    "person_income": "1",
    "person_emp_exp": "1",
    "loan_amnt": "1",
    "loan_int_rate": "0.01",
    "loan_percent_income": "0.1",
    "cb_person_cred_hist_length": "1",
    "credit_score": "1",
}


def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run `python scripts/train_model.py` first."
        )

    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


ARTIFACT = load_artifact()


def normalize_form_data(form_data):
    defaults = ARTIFACT["defaults"]
    normalized = defaults.copy()
    normalized.update(form_data.to_dict() if hasattr(form_data, "to_dict") else dict(form_data))

    for field in ARTIFACT["numeric_inputs"]:
        normalized[field] = float(normalized.get(field, defaults[field]))

    normalized["loan_percent_income"] = normalized["loan_percent_income"] / 100
    return normalized


def values_for_display(values):
    display_values = values.copy()
    display_values["loan_percent_income"] = round(
        float(display_values["loan_percent_income"]) * 100,
        2,
    )
    return display_values


def build_feature_frame(form_data):
    row = {}

    for field in ARTIFACT["numeric_inputs"]:
        row[field] = float(form_data[field])

    for field in ARTIFACT["one_hot_features"]:
        row[field] = form_data[field]

    binary_feature = ARTIFACT["binary_feature"]
    binary_value = form_data[binary_feature]
    row[binary_feature] = ARTIFACT["binary_mapping"][binary_value]

    input_df = pd.DataFrame([row])
    input_df = pd.get_dummies(
        input_df,
        columns=ARTIFACT["one_hot_features"],
        drop_first=True,
    )
    return input_df.reindex(columns=ARTIFACT["feature_columns"], fill_value=0)


def calculate_risk_score(form_data):
    risk_score = 50
    
    credit_score = float(form_data["credit_score"])
    loan_percent_income = float(form_data["loan_percent_income"])
    interest_rate = float(form_data["loan_int_rate"])
    previous_defaults = form_data["previous_loan_defaults_on_file"]
    employment = float(form_data["person_emp_exp"])
    
    if credit_score >= 750:
        risk_score -= 25
    elif credit_score >= 700:
        risk_score -= 18
    elif credit_score >= 650:
        risk_score -= 10
    elif credit_score >= 600:
        risk_score -= 3
    elif credit_score < 550:
        risk_score += 20
    
    if loan_percent_income <= 0.15:
        risk_score -= 15
    elif loan_percent_income <= 0.25:
        risk_score -= 8
    elif loan_percent_income > 0.40:
        risk_score += 15
    elif loan_percent_income > 0.35:
        risk_score += 10
    
    if interest_rate <= 8:
        risk_score -= 8
    elif interest_rate >= 18:
        risk_score += 12
    
    if employment >= 10:
        risk_score -= 12
    elif employment >= 5:
        risk_score -= 6
    elif employment <= 1:
        risk_score += 10
    
    if previous_defaults == "Yes":
        risk_score += 25

    return max(0, min(100, risk_score))


def get_risk_category(risk_score):
    if risk_score >= 75:
        return {"category": "HIGH", "color": "#b42318"}
    if risk_score >= 50:
        return {"category": "MEDIUM", "color": "#f59e0b"}
    return {"category": "LOW", "color": "#10b981"}


def explain_prediction(form_data, prediction, probability):
    credit_score = float(form_data["credit_score"])
    loan_percent_income = float(form_data["loan_percent_income"])
    interest_rate = float(form_data["loan_int_rate"])
    income = float(form_data["person_income"])
    loan_amount = float(form_data["loan_amnt"])
    previous_defaults = form_data["previous_loan_defaults_on_file"]
    employment = float(form_data["person_emp_exp"])

    risk_reasons = []
    support_reasons = []

    if previous_defaults == "Yes":
        risk_reasons.append("previous loan default is on file")
    else:
        support_reasons.append("no previous loan default is on file")

    if credit_score < 600:
        risk_reasons.append("credit score is below 600")
    elif credit_score >= 700:
        support_reasons.append("credit score is strong")

    if loan_percent_income >= 0.35:
        risk_reasons.append("loan request is more than 35% of annual income")
    elif loan_percent_income <= 0.20:
        support_reasons.append("loan request is 20% or less of annual income")

    if interest_rate >= 15:
        risk_reasons.append("interest rate is relatively high")
    elif interest_rate <= 10:
        support_reasons.append("interest rate is relatively low")

    if income > 0 and loan_amount / income >= 0.35:
        risk_reasons.append("loan-to-income ratio is high")

    if employment >= 5:
        support_reasons.append("employment experience is stable")
    elif employment <= 1:
        risk_reasons.append("employment experience is low")

    if prediction == 1:
        reasons = support_reasons[:3] or [
            "the model probability crossed the acceptance threshold"
        ]
        summary = "Accepted mainly because " + ", ".join(reasons) + "."
    else:
        reasons = risk_reasons[:3] or [
            "the model probability stayed below the acceptance threshold"
        ]
        summary = "Rejected mainly because " + ", ".join(reasons) + "."

    return {
        "summary": summary,
        "risk_reasons": risk_reasons,
        "support_reasons": support_reasons,
        "probability": probability,
    }


def predict_loan_status(form_data):
    normalized_data = normalize_form_data(form_data)
    features = build_feature_frame(normalized_data)
    probability = float(ARTIFACT["model"].predict_proba(features)[0, 1])
    prediction = int(probability >= ARTIFACT["threshold"])
    explanation = explain_prediction(normalized_data, prediction, probability)
    risk_score = calculate_risk_score(normalized_data)
    risk_category = get_risk_category(risk_score)
    return prediction, probability, explanation, risk_score, risk_category, normalized_data


@app.route("/", methods=["GET", "POST"])
def index():
    values = values_for_display(ARTIFACT["defaults"])
    result = None

    if request.method == "POST":
        prediction, probability, explanation, risk_score, risk_category, normalized_data = (
            predict_loan_status(request.form)
        )
        values = values_for_display(normalized_data)
        accepted = prediction == 1

        result = {
            "status_code": prediction,
            "status_label": "Accepted" if accepted else "Rejected",
            "title": "Loan likely accepted" if accepted else "Loan likely rejected",
            "message": explanation["summary"],
            "probability": probability,
            "class_name": "accepted" if accepted else "rejected",
            "risk_reasons": explanation["risk_reasons"],
            "support_reasons": explanation["support_reasons"],
            "risk_score": risk_score,
            "risk_category": risk_category,
        }

    return render_template(
        "index.html",
        values=values,
        result=result,
        labels=LABELS,
        help_text=HELP_TEXT,
        option_labels=OPTION_LABELS,
        steps=STEPS,
        options=ARTIFACT["options"],
        metrics=ARTIFACT["metrics"],
        threshold=ARTIFACT["threshold"],
        numeric_inputs=ARTIFACT["numeric_inputs"],
        one_hot_features=ARTIFACT["one_hot_features"],
        binary_feature=ARTIFACT["binary_feature"],
    )


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host="127.0.0.1", port=5000, debug=debug, use_reloader=debug)
