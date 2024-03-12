# Customer_Churn_dvc
- conda activate C:\Users\pachp\Desktop\projects\Customer_Churn_dvc\churn1


#### create logger.py
#### create exception handling in exception.py
#### create data_ingestion
#### track data by dvc --> 

- dvc init 
- delete artifacts 
- git add . 
- git status 
- git commit -m "git untrack" 
- git push origin main 
- python main.py 
- dvc add artifacts/raw.csv 
- git add . 
- git commit -m "dvc track" 
- git push. 
- git logs 
- git checkout adddcommitnumber.

#### mlflow tracking
- export MLFLOW_TRACKING_URI=https://dagshub.com/pachpandemahesh300/Customer_Churn_dvc.mlflow \
- export MLFLOW_TRACKING_USERNAME=pachpandemahesh300 \
- export MLFLOW_TRACKING_PASSWORD=9f73334e82cb0e1dc7693e6eed2b0827c86e8db9 \
- python script.py

#### 