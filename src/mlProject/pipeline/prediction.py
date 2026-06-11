import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from mlProject.components.data_transformation import NUMERIC_FEATURES
from mlProject.config.configuration import ConfigurationManager
from mlProject.utils.common import load_env_file
from mlProject.utils.model_registry import get_production_model_path, load_registry
from mlProject import logger

class PredictionPipeline:
    def __init__(self, model_path: Path = None):
        self.unified_pipeline = None
        self._model_path = model_path
        if model_path is None:
            load_env_file()
            try:
                config_manager = ConfigurationManager()
                registry_config = config_manager.get_model_registry_config()
                self._model_path = registry_config.registry_path.parent / "model.joblib"
                if not self._model_path.exists():
                    model_eval_config = config_manager.get_model_evaluation_config()
                    self._model_path = model_eval_config.model_path
            except Exception:
                self._model_path = Path('artifacts/model_trainer/model.joblib')

    def predict(self, data):
        if self.unified_pipeline is None:
            model_path = self._model_path or Path('artifacts/model_trainer/model.joblib')
            from mlProject.utils.common import verify_model_integrity
            checksum_path = Path(str(model_path) + ".sha256")
            if not verify_model_integrity(model_path, checksum_path):
                raise ValueError(f"Model integrity check failed for {model_path}")
            self.unified_pipeline = joblib.load(model_path)
            logger.info(f"Loaded unified pipeline from {model_path}")

        # Convert input to appropriate format
        if isinstance(data, np.ndarray):
            input_data = data
        elif isinstance(data, pd.DataFrame):
            try:
                input_data = data[NUMERIC_FEATURES].values
            except (KeyError, ValueError):
                input_data = data.values
        else:
            input_data = data

        # Unified pipeline handles preprocessing (if available) and prediction
        try:
            prediction = self.unified_pipeline.predict(input_data)
        except TypeError as e:
            # If unified_pipeline is just a model without preprocessor
            logger.warning(f"Pipeline prediction failed: {e}. Attempting direct model prediction.")
            prediction = self.unified_pipeline.predict(input_data)
        
        return prediction
