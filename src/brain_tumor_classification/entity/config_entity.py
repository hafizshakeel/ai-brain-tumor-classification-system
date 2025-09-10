from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_model_path: Path
    model_name: str
    image_size: list
    num_classes: int
    learning_rate: float
    weight_decay: float
    freeze_base: bool

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    train_data_dir: Path
    val_split: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: str
    step_size: int
    gamma: float

@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    trained_model_path: Path
    test_data_dir: Path
    report: Path
    batch_size: int
    log_with_mlflow: bool
    mlflow_experiment: str
    mlflow_tracking_uri: str