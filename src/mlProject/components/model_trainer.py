import pandas as pd
import os
from mlProject import logger
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
from mlProject.entity.config_entity import ModelTrainerConfig
from mlProject.utils.model_registry import (
    get_version_id, compute_file_hash, register_model,
)

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

        version_id = get_version_id()
        model_filename = f"model_{version_id}.joblib"
        model_path_str = os.path.join(self.config.root_dir, model_filename)
        try:
            joblib.dump(lr, model_path_str)
            checksum_path = model_path_str + ".sha256"
            from mlProject.utils.common import save_checksum
            save_checksum(Path(model_path_str), Path(checksum_path))
        except Exception as e:
            logger.exception(f"Failed to save model to {model_path_str}")
            raise

        model_path = Path(model_path_str)
        data_hash = None
        try:
            data_hash = compute_file_hash(Path(self.config.train_data_path))
        except Exception as e:
            logger.warning(f"Could not compute data hash: {e}")

        params = {
            "alpha": self.config.alpha,
            "l1_ratio": self.config.l1_ratio,
        }

        registry_path = Path(self.config.root_dir).parent / "model_registry.json"
        try:
            register_model(
                registry_path=registry_path,
                model_path=model_path,
                version_id=version_id,
                metrics={},
                params=params,
                data_hash=data_hash,
            )
        except ValueError as e:
            logger.error(f"Model registry rejected version {version_id}: {e}")
        except Exception as e:
            logger.warning(f"Failed to register model in registry: {e}")

        stable_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(lr, stable_path)

        logger.info(f"Model {version_id} trained and saved to {stable_path}")
        logger.info(f"Train X shape: {train_x.shape}, Test X shape: {test_x.shape}")

        