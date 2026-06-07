import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import numpy as np
import joblib
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    


    def save_results(self):
        try:
            test_data = pd.read_csv(self.config.test_data_path)
        except FileNotFoundError:
            logger.error(f"Test data file not found: {self.config.test_data_path}")
            raise
        except Exception as e:
            logger.exception("Failed to load test data")
            raise

        try:
            model = joblib.load(self.config.model_path)
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.config.model_path}")
            raise
        except Exception as e:
            logger.exception("Failed to load model")
            raise

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        try:
            predicted_qualities = model.predict(test_x)
        except Exception as e:
            logger.exception("Model prediction failed")
            raise

        (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
        
        scores = {"rmse": rmse, "mae": mae, "r2": r2}
        save_json(path=Path(self.config.metric_file_name), data=scores)
        logger.info(f"Evaluation metrics saved: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")