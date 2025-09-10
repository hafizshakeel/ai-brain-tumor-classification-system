import torch
from src.brain_tumor_classification import logger
from src.brain_tumor_classification.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.brain_tumor_classification.pipeline.prep_base_model_pipeline import PrepBaseModelTrainingPipeline
from src.brain_tumor_classification.pipeline.model_training_pipeline import ModelTrainingPipeline
from src.brain_tumor_classification.pipeline.model_eval_pipeline import ModelEvalPipeline

import os, mlflow, sys

def main():
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "https://dagshub.com/hafizshakeel/brain_tumor_classification.mlflow"))
        
        # step 1. data ingestion pipeline 
        try:
            logger.info(f"{'='*20} DATA INGESTION {'='*20}")
            obj = DataIngestionTrainingPipeline()
            obj.main()
            logger.info(f"{'='*20} DATA INGESTION COMPLETED {'='*20}")
        except Exception as e:
            logger.exception(f"ERROR IN DATA INGESTION: {e}")
            raise e

        # step 2. prepare base model pipeline
        try:
            logger.info(f"{'='*20} PREPARE BASE MODEL {'='*20}")
            obj = PrepBaseModelTrainingPipeline()
            obj.main()
            logger.info(f"{'='*20} BASE MODEL READY {'='*20}")
        except Exception as e:
            logger.exception(f"ERROR IN BASE MODEL PREPARATION: {e}")
            raise e

        # step 3: training pipeline
        try:
            logger.info(f"{'='*20} MODEL TRAINING {'='*20}")
            obj = ModelTrainingPipeline()
            obj.main()
            logger.info(f"{'='*20} TRAINING COMPLETED {'='*20}")
        except Exception as e:
            logger.exception(f"ERROR IN MODEL TRAINING: {e}")
            raise e

        # step 4: evaluation pipeline
        try:
            logger.info(f"{'='*20} MODEL EVALUATION {'='*20}")
            obj = ModelEvalPipeline()
            obj.main()
            logger.info(f"{'='*20} EVALUATION COMPLETED {'='*20}")
        except Exception as e:
            logger.exception(f"ERROR IN MODEL EVALUATION: {e}")
            raise e
            
        logger.info(f"{'='*20} PIPELINE COMPLETED SUCCESSFULLY {'='*20}")
        return 0
            
    except Exception as e:
        logger.exception(f"PIPELINE FAILED: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())


