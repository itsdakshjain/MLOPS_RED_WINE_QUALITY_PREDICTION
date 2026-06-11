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

        # Load preprocessor if available (from data_transformation stage)
        preprocessor = None
        preprocessor_path = Path('artifacts/data_transformation/preprocessor.joblib')
        if preprocessor_path.exists():
            try:
                preprocessor = joblib.load(preprocessor_path)
                logger.info(f"Loaded preprocessor from {preprocessor_path}")
            except Exception as e:
                logger.warning(f"Failed to load preprocessor: {e}. Training model without preprocessor.")
        
        # Create unified pipeline: preprocessor + model
        if preprocessor is not None:
            # Preprocess training data using the loaded preprocessor
            train_x_transformed = preprocessor.transform(train_x)
            test_x_transformed = preprocessor.transform(test_x)
            
            # Train model on transformed data
            try:
                lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
                lr.fit(train_x_transformed, train_y)
            except Exception as e:
                logger.exception("Failed to train model")
                raise
            
            # Create unified pipeline for inference
            unified_pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", lr),
            ])
            logger.info("Created unified pipeline: preprocessor + model")
        else:
            # Train model directly on raw data if no preprocessor
            try:
                lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
                lr.fit(train_x, train_y)
                unified_pipeline = lr
            except Exception as e:
                logger.exception("Failed to train model")
                raise

        version_id = get_version_id()
        model_filename = f"model_{version_id}.joblib"
        model_path_str = os.path.join(self.config.root_dir, model_filename)
        try:
            joblib.dump(unified_pipeline, model_path_str)
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
        joblib.dump(unified_pipeline, stable_path)

        logger.info(f"Unified pipeline (preprocessor + model) {version_id} trained and saved to {stable_path}")
        logger.info(f"Train X shape: {train_x.shape}, Test X shape: {test_x.shape}")

        
