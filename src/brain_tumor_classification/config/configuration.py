from src.brain_tumor_classification.constants import *
from src.brain_tumor_classification.utils.common import read_yaml, create_directories
from src.brain_tumor_classification.entity.config_entity import DataIngestionConfig 
from src.brain_tumor_classification.entity.config_entity import PrepareBaseModelConfig
from src.brain_tumor_classification.entity.config_entity import TrainingConfig
from src.brain_tumor_classification.entity.config_entity import EvaluationConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self, source_type: str) -> DataIngestionConfig:
        """
        source_type: 'gdrive' or 'kaggle'
        """
        if source_type == "gdrive":
            config = self.config.data_ingestion_gdrive
        elif source_type == "kaggle":
            config = self.config.data_ingestion_kaggle
        else:
            raise ValueError(f"Invalid source_type: {source_type}, choose 'gdrive' or 'kaggle'")

        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
    

    def prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params.base_model
        create_directories([config.root_dir])

        return PrepareBaseModelConfig(
            root_dir=config.root_dir,
            base_model_path=config.base_model_path,
            updated_model_path=config.updated_model_path,
            model_name=params.model_name,
            image_size=params.image_size,
            num_classes=params.num_classes,
            learning_rate=params.learning_rate,
            weight_decay=params.weight_decay,
            freeze_base=params.freeze_base
        )
    
    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        params = self.params.training

        create_directories([config.root_dir])

        training_config = TrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            train_data_dir=Path(config.train_data_dir),
            val_split=params.val_split,
            epochs=params.epochs,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            weight_decay=params.weight_decay,
            optimizer=params.optimizer,
            scheduler=params.scheduler,
            step_size=params.step_size,
            gamma=params.gamma
        )
        return training_config


    def get_mlflow_config(self):
        return self.config.mlflow

    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation
        params = self.params.evaluation
        mlflow_config = self.config.mlflow

        create_directories([config.root_dir])

        evaluation_config = EvaluationConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(self.config.training.trained_model_path),
            test_data_dir=Path(config.test_data_dir),
            report=Path(config.report),
            batch_size=params.batch_size,
            log_with_mlflow=mlflow_config.log_with_mlflow,
            mlflow_experiment=mlflow_config.mlflow_experiment,
            mlflow_tracking_uri=mlflow_config.mlflow_tracking_uri
        )
        return evaluation_config