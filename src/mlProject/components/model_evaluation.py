import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import numpy as np
import joblib
from mlProject import logger
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from mlProject.utils.model_registry import load_registry, save_registry
from mlProject.components.data_transformation import NUMERIC_FEATURES
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def baseline_r2_score(self, actual):
        """
        Calculate the R² score for a naive baseline model that always predicts the mean.
        R² < 0 indicates the model performs worse than predicting the mean.
        """
        mean_pred = np.full_like(actual, np.mean(actual))
        baseline_r2 = r2_score(actual, mean_pred)
        return baseline_r2

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
            from mlProject.utils.common import verify_model_integrity
            model_path = self.config.model_path
            checksum_path = Path(str(model_path) + ".sha256")
            if not verify_model_integrity(model_path, checksum_path):
                raise ValueError(f"Model integrity check failed for {model_path}")
            model = joblib.load(model_path)
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.config.model_path}")
            raise
        except Exception as e:
            logger.exception("Failed to load model")
            raise

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        preprocessor = None
        if self.config.preprocessor_path and self.config.preprocessor_path.exists():
            try:
                preprocessor = joblib.load(self.config.preprocessor_path)
                logger.info(f"Preprocessor loaded from {self.config.preprocessor_path}")
            except Exception as e:
                logger.warning(f"Could not load preprocessor: {e}")

        if preprocessor is not None:
            expected_cols = len(NUMERIC_FEATURES)
            if test_x.shape[1] != expected_cols:
                logger.warning(
                    f"test_x has {test_x.shape[1]} columns but preprocessor "
                    f"expects {expected_cols}. Selecting NUMERIC_FEATURES."
                )
                test_x = test_x[NUMERIC_FEATURES]
            test_x = preprocessor.transform(test_x)
        
        try:
            predicted_qualities = model.predict(test_x)
        except Exception as e:
            logger.exception("Model prediction failed")
            raise

        (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
        
        # Calculate baseline R² (predict-mean strategy)
        baseline_r2 = self.baseline_r2_score(test_y.values.flatten())
        
        # Validate model performance against baseline
        if r2 < 0.0:
            logger.error(f"Model R²={r2:.4f} is below baseline (R²=0.0). Aborting deployment.")
            raise ValueError(f"Model R²={r2:.4f} is below baseline. Aborting.")
        
        scores = {
            "rmse": rmse, 
            "mae": mae, 
            "r2": r2,
            "baseline_r2": baseline_r2
        }
        save_json(path=Path(self.config.metric_file_name), data=scores)

        logger.info(f"Evaluation metrics saved: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, Baseline R2={baseline_r2:.4f}")

        registry_path = self.config.root_dir.parent / "model_registry.json"
        self._update_registry_with_metrics(registry_path, scores)

        previous_metrics = self._load_previous_metrics(registry_path)
        if previous_metrics:
            comparison = self._compare_metrics(scores, previous_metrics)
            comparison_path = self.config.root_dir / "metrics_comparison.json"
            save_json(path=Path(comparison_path), data=comparison)
            logger.info(f"Metrics comparison saved to {comparison_path}")

    def _load_previous_metrics(self, registry_path: Path):
        """Load metrics from the previous production model."""
        registry = load_registry(registry_path)
        production_id = registry.get("production")
        if not production_id:
            return None
        for v in registry.get("versions", []):
            if v.get("id") == production_id and v.get("metrics"):
                return v["metrics"]
        return None

    def _compare_metrics(self, current: dict, previous: dict) -> dict:
        """Compare current metrics against previous."""
        comparison = {"current": current, "previous": previous, "changes": {}}
        for key in current:
            if key in previous and previous[key] != 0:
                pct_change = ((current[key] - previous[key]) / abs(previous[key])) * 100
                comparison["changes"][key] = round(pct_change, 2)
        return comparison

    def _update_registry_with_metrics(self, registry_path: Path, metrics: dict):
        """Update the latest model version in the registry with computed metrics."""
        try:
            registry = load_registry(registry_path)
            if registry["versions"]:
                registry["versions"][0]["metrics"] = metrics
                save_registry(registry_path, registry)
                logger.info(f"Updated registry with metrics: {metrics}")
        except Exception as e:
            logger.warning(f"Could not update registry with metrics: {e}")
