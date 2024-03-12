from src.customer_churn.logger import logging
from src.customer_churn.exception import CustomException
import sys
from src.customer_churn.components.data_ingestion import DataIngestion
from src.customer_churn.components.data_transformation import DataTransformation
from src.customer_churn.components.model_tranier import ModelTrainer
if __name__=="__main__":

    try:
        
        obj1=DataIngestion()
        train_data, test_data = obj1.initiate_data_ingestion()
        
        obj2=DataTransformation()
        train_arr, test_arr, _=obj2.initiate_data_transformation(train_data, test_data)

        obj3=ModelTrainer()
        obj3.initiate_model_trainer(train_arr, test_arr)

    except Exception as e:

        logging.info("Custom Exception")
        raise CustomException(e,sys)