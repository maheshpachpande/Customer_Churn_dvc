import sys
from dataclasses import dataclass
import numpy as np
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.combine import SMOTEENN

from src.customer_churn.exception import CustomException
from src.customer_churn.logger import logging
from src.customer_churn.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.datatransformationconfig = DataTransformationConfig()

    def get_data_transformer_object(self):

        '''
        This is responcible for data transformation
        '''

        try:

            num_features = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'Churn Score',
       'CLTV']
            
            cat_features = ['City', 'Gender', 'Senior Citizen', 'Partner', 'Dependents',
       'Phone Service', 'Multiple Lines', 'Internet Service',
       'Online Security', 'Online Backup', 'Device Protection', 'Tech Support',
       'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing',
       'Payment Method']
            
            logging.info("Pipeline initiated.....")
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer()),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(sparse_output=False, drop="first",dtype=np.int16))
                ]
            )

            logging.info("Pipeline completed.")

            logging.info("columntransformation initiated.....")
            preprocessor = ColumnTransformer(
                [
                    ("Numerical Pipeline", num_pipeline, num_features),
                    ("Categorical Pipeline", cat_pipeline, cat_features)
                ]
            )

            logging.info("columntransformation completed.")

            return preprocessor

        except Exception as e:
            logging.info("Error occured at data tranformation.....")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_set, test_set):
        try:
            train_df = pd.read_csv(train_set)
            test_df = pd.read_csv(test_set)
            logging.info("Read train and test data completed.")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "Churn Value"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on train and test dataset")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path = self.datatransformationconfig.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.datatransformationconfig.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info("Error occured at initiate data transformation.....")
            raise CustomException(e, sys)