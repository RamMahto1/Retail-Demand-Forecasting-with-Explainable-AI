from src.logger import logging
from src.exception import CustomException
import sys
import pandas as pd
import os
from src.utils import load_object
from datetime import datetime

# Paths for preprocessor and model
TRANSFORMER_PATH = os.path.join('artifacts', 'preprocessor.pkl')
MODEL_PATH = os.path.join('artifacts', 'model.pkl')


def preprocess_input(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived date columns if 'Date' exists to match preprocessor features.
    """
    try:
        if 'Date' in input_data.columns:
            input_data['Date'] = pd.to_datetime(input_data['Date'])
            input_data['day'] = input_data['Date'].dt.day
            input_data['month'] = input_data['Date'].dt.month
            input_data['year'] = input_data['Date'].dt.year
            input_data['weekday'] = input_data['Date'].dt.weekday
            input_data['is_weekend'] = input_data['weekday'].isin([5,6]).astype(int)
        
        return input_data
    
    except Exception as e:
        raise CustomException(e, sys)


def predict_test(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict Units_Sold for input DataFrame using trained preprocessor and model.

    Args:
        input_data (pd.DataFrame): Input data containing all required features.

    Returns:
        pd.DataFrame: Original data with 'Predicted_Units_Sold' column.
    """
    try:
        logging.info("Preprocessing input data")
        input_data = preprocess_input(input_data)

        logging.info("Loading preprocessor and model objects")
        preprocessor = load_object(TRANSFORMER_PATH)
        model = load_object(MODEL_PATH)

        # Transform features and predict
        logging.info("Transforming input data")
        input_features = preprocessor.transform(input_data)

        logging.info("Making predictions")
        predictions = model.predict(input_features)

        input_data['Predicted_Units_Sold'] = predictions

        logging.info("Predictions added to dataframe successfully")
        # Only return key columns
        return input_data[['Store_ID', 'Product_ID', 'Date', 'Predicted_Units_Sold']]

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Example input
        input_data = pd.DataFrame(
            {
                'Store_ID': [101],
                'Product_ID': [1001],
                'Category': ['Electronics'],
                'Region': ['North'],
                'Weather_Condition': ['Sunny'],
                'Seasonality': ['Holiday'],
                'Inventory_Level': [500],
                'Units_Ordered': [50],
                'Price': [299.99],
                'Discount': [10.0],
                'Holiday_Promotion': [1],
                'Competitor_Pricing': [279.99],
                'Date': ['2023-11-25'],
                'Units_Sold_Lag1': [45],
                'Units_Sold_Lag7': [300]
            }
        )

        # Get predictions
        result_df = predict_test(input_data)
        print(result_df)

        # Save predictions
        result_df.to_csv(os.path.join('artifacts', 'predicted_test_data.csv'), index=False)
        logging.info("Predicted data saved successfully")

    except Exception as e:
        raise CustomException(e, sys)
