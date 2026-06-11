import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mlProject.components.data_transformation import DataTransformation, NUMERIC_FEATURES, FeatureEngineer, OutlierCapper
from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformationSplitTest(unittest.TestCase):
    def test_train_test_split_is_deterministic_and_stratified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            data_path = root_dir / "wine.csv"
            rows = []
            for quality in (5, 6):
                for i in range(20):
                    rows.append({
                        "fixed acidity": i, "volatile acidity": i/10, "citric acid": i/20,
                        "residual sugar": i, "chlorides": i/100, "free sulfur dioxide": i,
                        "total sulfur dioxide": i*2, "density": 0.99 + i/1000, "pH": 3.0 + i/100,
                        "sulphates": i/20, "alcohol": 9.0 + i/10, "quality": quality,
                    })
            pd.DataFrame(rows).to_csv(data_path, index=False)

            config = DataTransformationConfig(
                root_dir=root_dir,
                data_path=data_path,
                test_size=0.25,
                random_state=42,
                stratify_column="quality",
                use_scaler=False,
            )

            DataTransformation(config).train_test_spliting()
            first_train = pd.read_csv(root_dir / "train.csv")
            first_test = pd.read_csv(root_dir / "test.csv")

            DataTransformation(config).train_test_spliting()
            second_train = pd.read_csv(root_dir / "train.csv")
            second_test = pd.read_csv(root_dir / "test.csv")

            pd.testing.assert_frame_equal(first_train, second_train)
            pd.testing.assert_frame_equal(first_test, second_test)
            self.assertEqual(len(first_train), 30)
            self.assertEqual(len(first_test), 10)
            self.assertEqual(first_test["quality"].value_counts().to_dict(), {5: 5, 6: 5})


class TestPreprocessing(unittest.TestCase):
    def test_feature_engineer_adds_derived_features(self):
        data = np.array([
            [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4],
            [7.8, 0.88, 0.0, 2.6, 0.098, 25.0, 67.0, 0.9968, 3.20, 0.68, 9.8],
        ], dtype=float)
        fe = FeatureEngineer(
            add_acidity_index=True,
            add_alcohol_sugar_ratio=True,
            add_free_sulfur_pct=True,
        )
        transformed = fe.fit_transform(data)
        self.assertEqual(transformed.shape[1], 14)
        fe_names = fe.get_feature_names_out()
        self.assertIn("acidity_index", fe_names)
        self.assertIn("alcohol_sugar_ratio", fe_names)
        self.assertIn("free_sulfur_pct", fe_names)

    def test_derived_features_mathematically_correct(self):
        data = np.array([
            [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4],
            [7.8, 0.88, 0.0, 2.6, 0.098, 25.0, 67.0, 0.9968, 3.20, 0.68, 9.8],
        ], dtype=float)
        fe = FeatureEngineer(
            add_acidity_index=True,
            add_alcohol_sugar_ratio=True,
            add_free_sulfur_pct=True,
        )
        transformed = fe.fit_transform(data)
        # acidity_index = fixed_acidity / pH
        expected_acidity = data[:, 0] / data[:, 8]
        np.testing.assert_array_almost_equal(transformed[:, 11], expected_acidity)
        # alcohol_sugar_ratio = alcohol / residual_sugar
        expected_ratio = data[:, 10] / data[:, 3]
        np.testing.assert_array_almost_equal(transformed[:, 12], expected_ratio)
        # free_sulfur_pct = free_sulfur_dioxide / total_sulfur_dioxide * 100
        expected_pct = data[:, 5] / data[:, 6] * 100
        np.testing.assert_array_almost_equal(transformed[:, 13], expected_pct)

    def test_feature_engineer_raises_on_missing_column(self):
        fe = FeatureEngineer(add_acidity_index=True)
        with self.assertRaises(ValueError):
            fe.fit(np.array([[1.0, 2.0, 3.0]]))

    def test_outlier_capper_clips_extreme_values(self):
        data = np.array([[1.0, 1000.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=float)
        capper = OutlierCapper(method="iqr", iqr_multiplier=1.5)
        transformed = capper.fit_transform(data)
        self.assertLess(transformed[0, 1], 1000.0)

    def test_preprocessing_pipeline_saved_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            data_path = root_dir / "wine.csv"
            rows = []
            for quality in (5, 6):
                for i in range(20):
                    rows.append({
                        "fixed acidity": i, "volatile acidity": i/10, "citric acid": i/20,
                        "residual sugar": i, "chlorides": i/100, "free sulfur dioxide": i,
                        "total sulfur dioxide": i*2, "density": 0.99 + i/1000, "pH": 3.0 + i/100,
                        "sulphates": i/20, "alcohol": 9.0 + i/10, "quality": quality,
                    })
            pd.DataFrame(rows).to_csv(data_path, index=False)
            config = DataTransformationConfig(
                root_dir=root_dir,
                data_path=data_path,
                test_size=0.25,
                random_state=42,
                stratify_column="quality",
                use_scaler=True,
                scaler_type="standard",
            )
            dt = DataTransformation(config)
            dt.train_test_spliting()
            preprocessor_path = root_dir / "preprocessor.joblib"
            self.assertTrue(preprocessor_path.exists())
            preprocessor = joblib.load(preprocessor_path)
            self.assertIsNotNone(preprocessor)


if __name__ == "__main__":
    unittest.main()
