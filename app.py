from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load the model, scaler, and encoder
model = joblib.load(r"C:\Users\Awoleye\Documents\model 1.pkl")
scaler = joblib.load(r"C:\Users\Awoleye\Documents\scaler1_model.pkl")
encoder = joblib.load(r"C:\Users\Awoleye\Documents\marital_encoder1.pkl")


app = Flask(__name__)


def make_prediction(input_data: dict):
    import pandas as pd

    required_keys = ['AGE_IN_YEARS', 'INCOME', 'HOME_OWNER', 'COLLEGE_DEGREE', 'GOOD_CREDIT', 'Marital Status']
    for key in required_keys:
        if key not in input_data:
            raise ValueError(f"Missing required key: {key}")

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale the numerical data including encoded 'Marital Status'
    to_scale = ['AGE_IN_YEARS', 'INCOME', 'HOME_OWNER', 'COLLEGE_DEGREE', 'GOOD_CREDIT', 'Marital Status']
    input_df[to_scale] = scaler.transform(input_df[to_scale])

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]

    return {
        "Churn Status": "Churn" if prediction == 1 else "No Churn",
        "Probability": probability
    }

# Route for the web page
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None

    if request.method == 'POST':
        input_data = {
            'AGE_IN_YEARS': float(request.form['AGE_IN_YEARS']),
            'INCOME': float(request.form['INCOME']),
            'HOME_OWNER': int(request.form['HOME_OWNER']),
            'COLLEGE_DEGREE': int(request.form['COLLEGE_DEGREE']),
            'GOOD_CREDIT': int(request.form['GOOD_CREDIT']),
            'Marital Status': int(request.form['MARITAL_STATUS'])  # Already encoded in HTML
        }

        result = make_prediction(input_data)
        prediction = result["Churn Status"]
        probability = result["Probability"]

    return render_template('index.html', prediction=prediction, probability=probability)

# Start the app
if __name__ == '__main__':
    app.run(debug=True)

