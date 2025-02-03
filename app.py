from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and feature columns
model = joblib.load('car_price_model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form

        # Convert form data into dictionary
        input_data = {
            "Present_Price": float(data['Present_Price']),
            "Kms_Driven": int(data['Kms_Driven']),
            "Owner": int(data['Owner']),
            "Fuel_Type_Diesel": 1 if data['Fuel_Type'] == 'Diesel' else 0,
            "Fuel_Type_Petrol": 1 if data['Fuel_Type'] == 'Petrol' else 0,
            "Seller_Type_Individual": 1 if data['Seller_Type'] == 'Individual' else 0,
            "Transmission_Manual": 1 if data['Transmission'] == 'Manual' else 0,
            "no_year": int(data['no_year'])
        }

        # Convert to DataFrame and reindex columns
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_df)[0]

        return render_template('index.html', prediction_text=f'Estimated Car Price: ${prediction:.2f}')

    except Exception as e:
        return render_template('index.html', error_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
