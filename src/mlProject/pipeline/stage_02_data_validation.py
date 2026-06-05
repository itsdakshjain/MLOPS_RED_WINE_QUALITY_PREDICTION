from mlProject.components.data_validation import DataValidation
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger


STAGE_NAME = "Data Validation Stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()


if __name__ == "__main__":
    pipeline = DataValidationTrainingPipeline()
    pipeline.main()

        