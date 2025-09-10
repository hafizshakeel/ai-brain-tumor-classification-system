import torch

from src.brain_tumor_classification.config.configuration import ConfigurationManager
from src.brain_tumor_classification.components.model_trainer import TrainingPipeline
from src.brain_tumor_classification import logger


STATE_NAME = "model training"

class ModelTrainingPipeline:
    def __init__(self):
        pass
        
    def main(self):
        config_manager = ConfigurationManager()
        training_config = config_manager.get_training_config()
        mlflow_config = config_manager.get_mlflow_config()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        updated_model_path = config_manager.config["prepare_base_model"]["updated_model_path"]
        model = torch.load(updated_model_path, map_location=device, weights_only=False)
        model = model.to(device)

        trainer = TrainingPipeline(config=training_config, model=model, device=device, mlflow_config=mlflow_config)
        trainer.run()


if __name__ == "__main__":
    try:
        logger.info(f" >>> stage {STATE_NAME} stared \n")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f" >>> stage {STATE_NAME} completed \n")
    except Exception as e:
        logger.exception(e)
        raise e

