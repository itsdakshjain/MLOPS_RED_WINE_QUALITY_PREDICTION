from mlProject import logger
import pandas as pd
from mlProject.entity.config_entity import DataValidationConfig



class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config


    def validate_all_columns(self) -> bool:
        try:
            data = pd.read_csv(self.config.data_file)
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.config.data_file}")
            raise
        except Exception as e:
            logger.exception(f"Failed to read data file: {self.config.data_file}")
            raise

        try:
            schema_cols = dict(self.config.all_schema)
            all_errors = []

            for col in data.columns:
                if col not in schema_cols:
                    all_errors.append(f"Unexpected column '{col}' found in data")
                    continue
                expected_dtype = schema_cols[col]
                actual_dtype = str(data[col].dtype)
                if actual_dtype != expected_dtype:
                    all_errors.append(
                        f"Column '{col}' type mismatch: expected {expected_dtype}, got {actual_dtype}"
                    )
                null_count = data[col].isnull().sum()
                if null_count > 0:
                    all_errors.append(f"Column '{col}' has {null_count} null value(s)")

            for col in schema_cols:
                if col not in data.columns:
                    all_errors.append(f"Missing expected column '{col}' in data")

            validation_status = len(all_errors) == 0
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
                if all_errors:
                    f.write("\nErrors:\n")
                    for err in all_errors:
                        f.write(f"  - {err}\n")

            logger.info(f"Data validation {'passed' if validation_status else 'failed'} ({len(all_errors)} error(s))")
            return validation_status

        except Exception as e:
            logger.exception(f"Data validation failed with error: {e}")
            raise
