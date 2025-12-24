import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting the training and test dataset")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting": CatBoostRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "KNNRegressor": KNeighborsRegressor(),
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            best_model_score = max(model_report.values())#to get best model score   
            best_model_name = max(model_report, key=model_report.get)#to get best model name
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise Exception("No best model found")
            logging.info(f"Best model found {best_model_name} with accuracy score {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predcited=best_model.predict(X_test)
            r2=r2_score(y_test,predcited)   #to get r2 score
            logging.info(f"R2 score of the best model is {r2}") #to log r2 score

            logging.info("Best model found and saved.")
            return r2
        except Exception as e:
            raise CustomException(e, sys)   