import os
import sys
import dill
import pandas as pd
import pymysql
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from src.customer_churn.exception import CustomException
from src.customer_churn.logger import logging
from dotenv import load_dotenv




load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv('db')



def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established",mydb)
        df=pd.read_sql_query('select * from churn',mydb)
        print(df.head())

        return df



    except Exception as ex:
        raise CustomException(ex)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logging.info("Error occured at save object")
        raise CustomException(e, sys)
    


def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param=param[list(models.keys())[i]]

            gs = GridSearchCV(estimator=model, param_grid=param, cv=5, n_jobs=-1)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            test_model_accuracy_score=accuracy_score(y_test, y_pred.round())
            report[list(models.keys())[i]]=test_model_accuracy_score

        return report

    except Exception as e:
        logging.info("Error occured at evaluate model")
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)