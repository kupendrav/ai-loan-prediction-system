import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "loan_data.csv"
MODEL_PATH = BASE_DIR / "models" / "loan_status_new_model.pkl"
TARGET = "loan_status"

BINARY_FEATURE = "previous_loan_defaults_on_file"
ONE_HOT_FEATURES = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
]
OUTLIER_COLUMNS = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]
NUMERIC_INPUTS = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]


def remove_outliers_iqr(frame, columns, factor=1.5):
    cleaned = frame.copy()
    mask = pd.Series(True, index=cleaned.index)

    for col in columns:
        q1 = cleaned[col].quantile(0.25)
        q3 = cleaned[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask &= cleaned[col].between(lower, upper)

    return cleaned.loc[mask].copy()


def encode_training_data(raw_df):
    df = raw_df.copy()
    df[BINARY_FEATURE] = LabelEncoder().fit_transform(df[BINARY_FEATURE])
    df = pd.get_dummies(df, columns=ONE_HOT_FEATURES, drop_first=True)
    return df.astype(int)


def find_best_threshold(model, X_val, y_val):
    probabilities = model.predict_proba(X_val)[:, 1]
    best_threshold = 0.5
    best_accuracy = 0

    for threshold in np.arange(0.05, 0.96, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        accuracy = accuracy_score(y_val, predictions)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_threshold, best_accuracy


def main():
    raw_df = pd.read_csv(DATA_PATH)
    encoded_df = encode_training_data(raw_df)
    cleaned_df = remove_outliers_iqr(encoded_df, OUTLIER_COLUMNS)

    X = cleaned_df.drop(columns=[TARGET])
    y = cleaned_df[TARGET]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.2,
        random_state=42,
        stratify=y_trainval,
    )

    best_score = -1
    best_threshold = 0.5
    best_var_smoothing = None

    for var_smoothing in np.logspace(-12, -1, 12):
        candidate = GaussianNB(var_smoothing=var_smoothing)
        candidate.fit(X_train, y_train)
        threshold, validation_accuracy = find_best_threshold(candidate, X_val, y_val)

        if validation_accuracy > best_score:
            best_score = validation_accuracy
            best_threshold = threshold
            best_var_smoothing = float(var_smoothing)

    new_model = GaussianNB(var_smoothing=best_var_smoothing)
    new_model.fit(X_trainval, y_trainval)

    test_probabilities = new_model.predict_proba(X_test)[:, 1]
    y_pred_new = (test_probabilities >= best_threshold).astype(int)
    test_accuracy = accuracy_score(y_test, y_pred_new)

    artifact = {
        "model": new_model,
        "threshold": best_threshold,
        "feature_columns": X.columns.tolist(),
        "numeric_inputs": NUMERIC_INPUTS,
        "one_hot_features": ONE_HOT_FEATURES,
        "binary_feature": BINARY_FEATURE,
        "binary_mapping": {"No": 0, "Yes": 1},
        "target": TARGET,
        "options": {
            col: sorted(raw_df[col].dropna().unique().tolist())
            for col in ONE_HOT_FEATURES + [BINARY_FEATURE]
        },
        "defaults": {
            "person_age": 30,
            "person_gender": "male",
            "person_education": "Bachelor",
            "person_income": 60000,
            "person_emp_exp": 5,
            "person_home_ownership": "RENT",
            "loan_amnt": 10000,
            "loan_intent": "PERSONAL",
            "loan_int_rate": 12.0,
            "loan_percent_income": 0.17,
            "cb_person_cred_hist_length": 5,
            "credit_score": 650,
            "previous_loan_defaults_on_file": "No",
        },
        "metrics": {
            "original_rows": int(raw_df.shape[0]),
            "cleaned_rows": int(cleaned_df.shape[0]),
            "rows_removed": int(raw_df.shape[0] - cleaned_df.shape[0]),
            "validation_accuracy": float(best_score),
            "test_accuracy": float(test_accuracy),
            "var_smoothing": best_var_smoothing,
            "threshold": best_threshold,
            "confusion_matrix": confusion_matrix(y_test, y_pred_new).tolist(),
            "classification_report": classification_report(
                y_test,
                y_pred_new,
                output_dict=True,
            ),
        },
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as file:
        pickle.dump(artifact, file)

    print(f"Saved {MODEL_PATH}")
    print(f"Cleaned rows: {cleaned_df.shape[0]} / {raw_df.shape[0]}")
    print(f"Validation accuracy: {best_score:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Var smoothing: {best_var_smoothing}")


if __name__ == "__main__":
    main()
