from src.customer_churn.logger import logging
from src.customer_churn.exception import CustomException
import sys
from src.customer_churn.components.data_ingestion import DataIngestion

if __name__=="__main__":

    try:
        
        obj=DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        print(train_data)

    except Exception as e:
        
        logging.info("Custom Exception")
        raise CustomException(e,sys)