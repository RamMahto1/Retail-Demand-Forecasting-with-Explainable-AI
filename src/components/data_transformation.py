from src.logger import logging
from src.exception import CustomException
import sys
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            numerical_cols = ['Inventory_Level', 'Units_Ordered', 'Price', 'Discount', 'Holiday_Promotion','Competitor_Pricing', 'day', 'month', 'year', 'weekday', 
                              'is_weekend', 'Units_Sold_Lag1', 'Units_Sold_Lag7']
            categorical_cols = ['Store_ID', 'Product_ID', 'Category', 'Region', 'Weather_Condition', 'Seasonality']

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])
            
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])
            logging.info(f"Numerical columns: {numerical_cols}")
            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info("Preprocessor object created")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data")
            
            logging.info("Obtaining preprocessor object")
            preprocessor = self.get_data_transformer_object()
            
            target_column_name = 'Units_Sold'
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training and testing data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info("Saved preprocessing object")
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            raise CustomException(e, sys)