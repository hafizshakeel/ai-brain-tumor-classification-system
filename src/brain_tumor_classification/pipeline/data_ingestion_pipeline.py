from src.brain_tumor_classification.config.configuration import ConfigurationManager
from src.brain_tumor_classification.components.data_ingestion import DataIngestion
from src.brain_tumor_classification import logger

STATE_NAME = "data ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
        
    def main(self):
        config = ConfigurationManager()
        # Source: kaggle or gdrive
        data_ingestion_cfg = config.get_data_ingestion_config("kaggle")
        data_ingestion = DataIngestion(config=data_ingestion_cfg, source="kaggle")
        data_ingestion.download()
        data_ingestion.extract_zip_file()

if __name__ == "__main__":
    try:
        logger.info(f" >>> stage {STATE_NAME} stared \n")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f" >>> stage {STATE_NAME} completed \n")
    except Exception as e:
        logger.exception(e)
        raise e

