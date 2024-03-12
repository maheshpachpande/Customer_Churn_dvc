import os 
import sys
import numpy as np
import pandas as pd
import mlflow
from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.customer_churn.exception import CustomException
from src.customer_churn.logger import logging
from src.customer_churn.utils import save_object, evaluate_model, eval_metrics
from imblearn.combine import SMOTEENN
from urllib.parse import urlparse


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split the data into Train and Test...........")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            sme = SMOTEENN(random_state=42)
            x_res, y_res = sme.fit_resample(X_train, y_train)

            params={
                # 'RandomForestClassifier':{
                #      'criterion':['gini', 'entropy', 'log_loss'],                 
                #      'max_features':['auto','sqrt','log2',None],
                #      'max_depth':[int(x) for x in np.linspace(10, 1000, 10)],
                #      'min_samples_split':[1,3,4,5,7,8,9,10],
                #      'n_estimators': #[8,16,32,64,128,256,512,1024,2048] ,
                #             [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
                #      'min_samples_leaf':[1,2,3,4,5,6,7,8]
                #  },
                # 'GradientBoostingClassifier':{
                #     'loss':['log_loss', 'exponential'],
                #     'learning_rate':[.1,.01,.05,.001],
                #     'n_estimators': [8,16,32,64,128,256,512,1024],
                #     'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                #     'criterion':['squared_error', 'friedman_mse'],
                #     'max_features':['auto','sqrt','log2'],
                #     },
                # 'LogisticRegression':{
                #     'max_iter':[100, 200, 300]
                #     },
                # 'XGBClassifier':{
                #     'eval_metric':['auc','logloss','error'],
                #     'eta':[0.01,0.02,0.05, 0.08, 0.1, 0.15, 0.19],
                #     'max_depth':[3,4,5,6,7,8,9,10],
                #     'subsample':[0.5,0.6,0.7,0.8,0.9]
                #     },
                # 'CatBoostClassifier':{},
                # 'AdaBoostClassifier':{                    
                #     'n_estimators': [8,16,32,64,128,256,512,1024]
                #     },
                'KNeighborsClassifier':{
                    # 'n_neighbors':[1,2,3,4,5,6,7,8,9],
                    'weights':['uniform','distance'],
                    # 'algorithm':['auto', 'ball_tree','kd_tree']
                    }
                
            }

            models={
                # 'LogisticRegression':LogisticRegression(),
                  'KNeighborsClassifier':KNeighborsClassifier(),
            #     'XGBClassifier':XGBClassifier(),
            #     'CatBoostClassifier':CatBoostClassifier(verbose=True),
            #     'AdaBoostClassifier':AdaBoostClassifier(),
            #     'GradientBoostingClassifier':GradientBoostingClassifier(),
            #     'RandomForestClassifier':RandomForestClassifier()
             }


            model_report:dict=evaluate_model(X_train=x_res, y_train=y_res, X_test=X_test, 
                                             y_test=y_test, models=models, param=params)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print("This is the Best Model: ")
            print(best_model_name)

            model_names = list(params.keys())

            actual_model = ""

            for model in model_names:
                if best_model_name==model:
                    actual_model=actual_model+model

            best_params = params[actual_model]

            mlflow.set_registry_uri("https://dagshub.com/pachpandemahesh300/Customer_Churn_dvc.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            #mlFlow
            
            with mlflow.start_run():

                predicted_qualities = best_model.predict(X_test)

                accuracy, class_report, confusionmatrix = eval_metrics(y_test, predicted_qualities)

                mlflow.log_params(best_params)

                mlflow.log_metric("accuracy", accuracy)
                # mlflow.log_metric("class_report", class_report)
                # mlflow.log_metric("confusionmatrix", confusionmatrix)

                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")

            if best_model_score < 0.6:
                raise CustomException("No Best Model Found......")
            
            logging.info("Best found model on both train and test dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy_score1 = accuracy_score(y_test, predicted)

            return accuracy_score1

        
        except Exception as e:
            logging.info("Error occured at initiate model training")
            raise CustomException(e, sys)