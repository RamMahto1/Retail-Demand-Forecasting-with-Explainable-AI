from src.logger import logging
from src.exception import CustomException
import sys
from datetime import datetime
from src.components.data_ingestion import DataIngestion

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_validation import DataValidation

try:
    
    ## Step 1: Data Ingestion
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    logging.info(f"Train data saved at: {train_data_path}")
    logging.info(f"Test data saved at: {test_data_path}")


    ## Step 2: Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    logging.info(f"Transformed Train data saved at: {train_arr}")
    logging.info(f"Transformed Test data saved at: {test_arr}")
    
    
    ## Step 3: Data Validation
    data_validation = DataValidation(train_data_path, test_data_path)
    data_validation.initiate_data_validation()
    logging.info(f"Data Validation completed successfully")

    ## Step 4: Model Trainer
    model_trainer = ModelTrainer()
    report,best_model_name,best_model,best_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    logging.info(f"Best Model: {best_model_name} with score: {best_score}")
except Exception as e:
    raise CustomException(e, sys)


logging.info(f"pipeline execute successfully")

    
    
    
    