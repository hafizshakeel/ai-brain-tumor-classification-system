from src.brain_tumor_classification.config.configuration import ConfigurationManager
from src.brain_tumor_classification.components.prep_base_model import PrepareBaseModel
from src.brain_tumor_classification import logger

STATE_NAME = "data ingestion stage"

class PrepBaseModelTrainingPipeline:
    def __init__(self):
        pass
        
    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)

        prepare_base_model.load_model()
        prepare_base_model.freeze_base_layers()
        prepare_base_model.save_updated_model()

if __name__ == "__main__":
    try:
        logger.info(f" >>> stage {STATE_NAME} stared \n")
        obj = PrepBaseModelTrainingPipeline()
        obj.main()
        logger.info(f" >>> stage {STATE_NAME} completed \n")
    except Exception as e:
        logger.exception(e)
        raise e

