import pandas as pd
import os
from mlProject import logger
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        try:
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)
        except FileNotFoundError as e:
            logger.error(f"Training data file not found: {e.filename}")
            raise
        except Exception as e:
            logger.exception("Failed to load training data")
            raise

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        try:
            lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
            lr.fit(train_x, train_y)
        except Exception as e:
            logger.exception("Failed to train model")
            raise

        try:
            joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))
        except Exception as e:
            logger.exception(f"Failed to save model to {self.config.model_name}")
            raise

        logger.info("Model training completed")
        logger.info(f"Train X shape: {train_x.shape}, Test X shape: {test_x.shape}")

        