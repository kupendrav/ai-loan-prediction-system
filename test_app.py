import unittest

from app import app


class LoanAppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_page_renders_form(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Loan Status Predictor", html)
        self.assertIn("Approval prediction with risk insights", html)
        self.assertIn("Predict Loan Status", html)
        self.assertIn("Loan as Percent of Income (%)", html)
        self.assertIn("No previous default", html)
        self.assertIn("Debt Consolidation", html)

    def test_prediction_renders_graphical_insights(self):
        values = {
            "person_age": "35",
            "person_gender": "male",
            "person_education": "Bachelor",
            "person_income": "72000",
            "person_emp_exp": "8",
            "person_home_ownership": "RENT",
            "loan_amnt": "12000",
            "loan_intent": "PERSONAL",
            "loan_int_rate": "11.5",
            "loan_percent_income": "0.17",
            "cb_person_cred_hist_length": "6",
            "credit_score": "710",
            "previous_loan_defaults_on_file": "No",
        }

        response = self.client.post("/", data=values)

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Risk Classification", html)
        self.assertIn("Prediction Analysis", html)
        self.assertIn("riskGaugeChart", html)
        self.assertIn("probabilityChart", html)
        self.assertIn("new Chart", html)
        self.assertIn("Positive signals", html)

    def test_loan_percent_income_accepts_percent_value(self):
        values = {
            "person_age": "25",
            "person_gender": "male",
            "person_education": "Doctorate",
            "person_income": "600007",
            "person_emp_exp": "5",
            "person_home_ownership": "OWN",
            "loan_amnt": "1000",
            "loan_intent": "EDUCATION",
            "loan_int_rate": "10",
            "loan_percent_income": "5",
            "cb_person_cred_hist_length": "10",
            "credit_score": "650",
            "previous_loan_defaults_on_file": "No",
        }

        response = self.client.post("/", data=values)

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('name="loan_percent_income"', html)
        self.assertIn('value="5.0"', html)
        self.assertNotIn("loan request is more than 35% of annual income", html)


if __name__ == "__main__":
    unittest.main()
