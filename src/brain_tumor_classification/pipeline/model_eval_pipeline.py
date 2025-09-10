import torch

from src.brain_tumor_classification.config.configuration import ConfigurationManager
from src.brain_tumor_classification.components.model_eval import EvaluationPipeline
from src.brain_tumor_classification import logger


STATE_NAME = "model training"

class ModelEvalPipeline:
    def __init__(self):
        pass
        
    def main(self):
        config_manager = ConfigurationManager()
        eval_config = config_manager.get_evaluation_config()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        evaluator = EvaluationPipeline(config=eval_config, device=device)
        report, cm = evaluator.run()


if __name__ == "__main__":
    try:
        logger.info(f" >>> stage {STATE_NAME} stared \n")
        obj = ModelEvalPipeline()
        obj.main()
        logger.info(f" >>> stage {STATE_NAME} completed \n")
    except Exception as e:
        logger.exception(e)
        raise e

