import os
import joblib
import numpy as np
from mlProject import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from mlProject.entity.config_entity import DataTransformationConfig


NUMERIC_FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
]


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Cap outliers using IQR-based capping or percentile-based capping."""
    def __init__(self, method: str = "iqr", iqr_multiplier: float = 1.5):
        self.method = method
        self.iqr_multiplier = iqr_multiplier
        self.lower_bounds = {}
        self.upper_bounds = {}

    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        for i in range(X_arr.shape[1]):
            col = X_arr[:, i]
            q1, q3 = np.percentile(col, [25, 75])
            iqr = q3 - q1
            self.lower_bounds[i] = q1 - self.iqr_multiplier * iqr
            self.upper_bounds[i] = q3 + self.iqr_multiplier * iqr
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        for i in range(X_arr.shape[1]):
            X_arr[:, i] = np.clip(X_arr[:, i], self.lower_bounds.get(i, -np.inf), self.upper_bounds.get(i, np.inf))
        return X_arr


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add derived features: acidity_index, alcohol_sugar_ratio, free_sulfur_pct."""
    def __init__(self, add_acidity_index=True, add_alcohol_sugar_ratio=True, add_free_sulfur_pct=True):
        self.add_acidity_index = add_acidity_index
        self.add_alcohol_sugar_ratio = add_alcohol_sugar_ratio
        self.add_free_sulfur_pct = add_free_sulfur_pct
        self.n_features_in_ = len(NUMERIC_FEATURES)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        additional = []
        if self.add_acidity_index and X_arr.shape[1] >= 9:
            fixed = X_arr[:, 0]
            ph = X_arr[:, 8]
            with np.errstate(divide='ignore', invalid='ignore'):
                idx = np.where(ph > 0, fixed / ph, 0)
            additional.append(idx)
        if self.add_alcohol_sugar_ratio and X_arr.shape[1] >= 11:
            alcohol = X_arr[:, 10]
            sugar = X_arr[:, 3]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(sugar > 0, alcohol / sugar, 0)
            additional.append(ratio)
        if self.add_free_sulfur_pct and X_arr.shape[1] >= 7:
            free_sulfur = X_arr[:, 5]
            total_sulfur = X_arr[:, 6]
            with np.errstate(divide='ignore', invalid='ignore'):
                pct = np.where(total_sulfur > 0, free_sulfur / total_sulfur * 100, 0)
            additional.append(pct)
        if additional:
            return np.column_stack([X_arr] + additional)
        return X_arr

    def get_feature_names_out(self, input_features=None):
        names = list(NUMERIC_FEATURES)
        if self.add_acidity_index:
            names.append("acidity_index")
        if self.add_alcohol_sugar_ratio:
            names.append("alcohol_sugar_ratio")
        if self.add_free_sulfur_pct:
            names.append("free_sulfur_pct")
        return names


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def _build_preprocessing_pipeline(self):
        """Build a full preprocessing pipeline with imputation, scaling, outlier capping, and feature engineering."""
        scaler_map = {
            "standard": StandardScaler(),
            "robust": RobustScaler(),
            "minmax": MinMaxScaler(),
        }
        scaler = scaler_map.get(self.config.scaler_type, StandardScaler())

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
            ("outlier_capper", OutlierCapper(method="iqr", iqr_multiplier=1.5)),
        ])

        preprocessor = Pipeline(steps=[
            ("numeric", numeric_transformer),
            ("feature_engineer", FeatureEngineer(
                add_acidity_index=True,
                add_alcohol_sugar_ratio=True,
                add_free_sulfur_pct=True,
            )),
        ])
        return preprocessor

    def train_test_spliting(self):
        try:
            data = pd.read_csv(self.config.data_path)
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.config.data_path}")
            raise
        except Exception as e:
            logger.exception(f"Failed to read data file: {self.config.data_path}")
            raise

        stratify = None
        if self.config.stratify_column:
            if self.config.stratify_column not in data.columns:
                raise ValueError(
                    f"Stratify column '{self.config.stratify_column}' "
                    "not found in transformed data"
                )
            stratify = data[self.config.stratify_column]

        try:
            train, test = train_test_split(
                data,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify,
            )
        except ValueError as exc:
            if stratify is None:
                logger.exception("train_test_split failed")
                raise
            logger.warning(
                "Falling back to non-stratified split because '%s' cannot be "
                "stratified safely: %s",
                self.config.stratify_column,
                exc,
            )
            train, test = train_test_split(
                data,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )

        if self.config.use_scaler:
            preprocessor = self._build_preprocessing_pipeline()
            train_features = train[NUMERIC_FEATURES]
            test_features = test[NUMERIC_FEATURES]
            train_target = train[[self.config.stratify_column]] if self.config.stratify_column in train.columns else None
            test_target = test[[self.config.stratify_column]] if self.config.stratify_column in test.columns else None

            train_scaled = preprocessor.fit_transform(train_features)
            test_scaled = preprocessor.transform(test_features)

            try:
                feat_names = preprocessor.named_steps["feature_engineer"].get_feature_names_out()
            except Exception:
                feat_names = [f"feat_{i}" for i in range(train_scaled.shape[1])]

            train_scaled_df = pd.DataFrame(train_scaled, columns=feat_names)
            test_scaled_df = pd.DataFrame(test_scaled, columns=feat_names)

            if train_target is not None:
                train_scaled_df[self.config.stratify_column] = train_target.values
                test_scaled_df[self.config.stratify_column] = test_target.values

            train_result = train_scaled_df
            test_result = test_scaled_df

            preprocessor_path = os.path.join(self.config.root_dir, "preprocessor.joblib")
            joblib.dump(preprocessor, preprocessor_path)
            logger.info(f"Preprocessing pipeline saved to {preprocessor_path}")
        else:
            train_result = train
            test_result = test

        try:
            train_result.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
            test_result.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
        except OSError as e:
            logger.error(f"Failed to write train/test CSV files: {e}")
            raise

        logger.info("Splited data into training and test sets")
        logger.info(train_result.shape)
        logger.info(test_result.shape)

        print(train_result.shape)
        print(test_result.shape)
