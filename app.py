from flask import Flask, request, render_template
import pandas as pd
import os
from datetime import datetime
from src.utils import load_object
from predict_test import preprocess_input, predict_test

app = Flask(__name__)

# Paths for artifacts
TRANSFORMER_PATH = os.path.join('artifacts', 'preprocessor.pkl')
MODEL_PATH = os.path.join('artifacts', 'model.pkl')


def validate_input(form_data):
    """
    Validate form inputs and convert to proper types.
    Returns: dict of cleaned data or raises ValueError
    """
    try:
        data = {
            'Store_ID': [int(form_data['Store_ID'])],
            'Product_ID': [int(form_data['Product_ID'])],
            'Category': [form_data['Category']],
            'Region': [form_data['Region']],
            'Weather_Condition': [form_data['Weather_Condition']],
            'Seasonality': [form_data['Seasonality']],
            'Inventory_Level': [float(form_data['Inventory_Level'])],
            'Units_Ordered': [int(form_data['Units_Ordered'])],
            'Price': [float(form_data['Price'])],
            'Discount': [float(form_data['Discount'])],
            'Holiday_Promotion': [int(form_data['Holiday_Promotion'])],
            'Competitor_Pricing': [float(form_data['Competitor_Pricing'])],
            'Date': [form_data['Date']],
            'Units_Sold_Lag1': [int(form_data['Units_Sold_Lag1'])],
            'Units_Sold_Lag7': [int(form_data['Units_Sold_Lag7'])]
        }

        # Validate date format
        try:
            datetime.strptime(data['Date'][0], '%Y-%m-%d')
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format.")

        return data

    except KeyError as e:
        raise ValueError(f"Missing input field: {e}")
    except ValueError as ve:
        raise ValueError(f"Invalid input: {ve}")


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error_message = None

    if request.method == 'POST':
        try:
            # Validate and clean input
            cleaned_data = validate_input(request.form)
            input_df = pd.DataFrame(cleaned_data)

            # Preprocess and predict
            input_df = preprocess_input(input_df)
            result_df = predict_test(input_df)
            prediction = round(result_df['Predicted_Units_Sold'].values[0], 2)

        except Exception as e:
            error_message = str(e)

    return render_template('index.html', prediction=prediction, error=error_message)


if __name__ == "__main__":
    app.run(debug=True)
