
# Retail Demand Forecasting Project

## Project Overview

This project predicts daily sales (units sold) for retail products using historical data and relevant features. It implements a full machine learning workflow, from preprocessing raw data to training multiple models, selecting the best model, and deploying predictions via a web interface. The project demonstrates an end-to-end solution for retail demand forecasting.

## Features

* Data Preprocessing: Handles missing values, scales numeric features, and encodes categorical features.
* Feature Engineering: Generates time-based features such as day, month, year, weekday, and weekend indicators from date information.
* Model Training: Evaluates multiple regression models including Linear Regression, Decision Tree, Random Forest, Gradient Boosting, SVR, Ridge, and Lasso. Hyperparameters are optimized using GridSearchCV.
* Prediction Interface: Provides a Flask-based web application to enter product and store details and receive predicted sales.
* Reusable Pipeline: Saves preprocessing objects and trained models for consistent predictions on new data.

## Installation

# Clone the repository
git clone <your-repo-link>
cd retail_demand_forecasting

# Set up virtual environment
python -m venv venv
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## How to Use

### Training and Evaluation


python main.py

This script reads training and testing datasets, applies preprocessing, trains multiple models, evaluates them, and saves the best-performing model and preprocessing object in the `artifacts/` folder.

### Running the Web Application

python app.py

Navigate to `http://127.0.0.1:5000/` in your browser. Fill in the product and store details to predict the number of units sold.

### Input Example

You can submit input either through the HTML form or via a JSON payload:

```json
{
  "Store_ID": 101,
  "Product_ID": 1001,
  "Category": "Electronics",
  "Region": "North",
  "Weather_Condition": "Sunny",
  "Seasonality": "Holiday",
  "Inventory_Level": 500,
  "Units_Ordered": 50,
  "Price": 299.99,
  "Discount": 10,
  "Holiday_Promotion": 1,
  "Competitor_Pricing": 279.99,
  "Date": "2023-11-25",
  "Units_Sold_Lag1": 45,
  "Units_Sold_Lag7": 300
}
```

## Project Structure

```
retail_demand_forecasting/
│
├── artifacts/          # Stores trained models and preprocessors
├── src/                # Python modules: utils, logger, exception
├── templates/          # HTML templates for the Flask app
├── main.py             # Script to train and evaluate models
├── app.py              # Flask web app
├── predict_test.py     # Testing script for predictions
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Tools & Technologies

* Python: Data manipulation and ML pipeline
* Pandas / NumPy: Data handling
* scikit-learn: Model training and evaluation
* Flask: Web app deployment
* GridSearchCV: Hyperparameter optimization
* HTML / Templates: User input forms

## Author

Ram – Data Scientist & Machine Learning Engineer


