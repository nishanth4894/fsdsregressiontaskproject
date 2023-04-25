# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting Dependent and Independent variables from train and test")
            X_train, y_train, X_test, y_test = (
                train_array[:,[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17]],
                train_array[:,-7],
                test_array[:,[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17]],
                test_array[:,-7]
            )

            ## Train multiple models

            models = {
            "LinearRegression":LinearRegression(),
            "Lasso":Lasso(),
            "Ridge":Ridge(),
            "Elasticnet":ElasticNet()
        }
            

            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print("\n=================================================================================")
            logging.info(f"Model Report : {model_report}")

            # To get the best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )

        except Exception as e:
            logging.info("Exception occured at Model training")
            raise CustomException(e,sys)