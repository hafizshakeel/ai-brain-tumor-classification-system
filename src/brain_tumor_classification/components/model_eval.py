import os, json, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import mlflow

from src.brain_tumor_classification import logger
from src.brain_tumor_classification.constants import *
from src.brain_tumor_classification.utils.common import read_yaml, create_directories


class EvaluationPipeline:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        # Data transforms
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # Dataset & dataloader
        self.test_dataset = datasets.ImageFolder(self.config.test_data_dir, transform=self.test_transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False)

        self.class_names = self.test_dataset.classes

        # Load model
        self.model = self._load_model()

    def _load_model(self):
        """Safely load the model with proper error handling"""
        try:
            # Try to load the entire model first
            model = torch.load(self.config.trained_model_path, map_location=self.device, weights_only=False)
            logger.info("Loaded entire model successfully")
            return model
        except Exception as e:
            logger.warning(f"Could not load entire model: {e}. Trying to load state dict...")
            try:
                # If that fails, try to load just the state dict
                # This assumes you know the model architecture
                from torchvision.models import swin_t
                model = swin_t(weights=None)
                num_ftrs = model.head.in_features
                model.head = torch.nn.Linear(num_ftrs, len(self.class_names))
                
                state_dict = torch.load(self.config.trained_model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info("Loaded model state dict successfully")
                return model
            except Exception as e2:
                logger.error(f"Could not load model state dict either: {e2}")
                raise

    def run(self):
        logger.info("Starting Evaluation...")
        
        # Set model to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        all_labels, all_preds = [], []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        # Compute metrics
        report_dict = classification_report(
            all_labels,
            all_preds,
            target_names=self.class_names,
            digits=4,
            output_dict=True,
            zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds)
        
        logger.info(f"\nClassification Report:\n{json.dumps(report_dict, indent=2)}")
        logger.info(f"Confusion Matrix:\n{cm}")

        # Save classification report
        os.makedirs(os.path.dirname(self.config.report), exist_ok=True)
        with open(self.config.report, "w") as f:
            json.dump(report_dict, f, indent=4)

        # MLflow logging
        if self.config.log_with_mlflow:
            try:
                mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
                mlflow.set_experiment(self.config.mlflow_experiment)
                
                with mlflow.start_run(run_name="evaluation_run") as run:
                    # Log parameters
                    mlflow.log_params({
                        "batch_size": self.config.batch_size,
                        "dataset": "test",
                        "model_path": str(self.config.trained_model_path)
                    })
                    
                    # Log overall metrics
                    mlflow.log_metrics({
                        "accuracy": report_dict["accuracy"],
                        "macro_precision": report_dict["macro avg"]["precision"],
                        "macro_recall": report_dict["macro avg"]["recall"],
                        "macro_f1": report_dict["macro avg"]["f1-score"],
                        "weighted_precision": report_dict["weighted avg"]["precision"],
                        "weighted_recall": report_dict["weighted avg"]["recall"],
                        "weighted_f1": report_dict["weighted avg"]["f1-score"],
                    })
                    
                    # Log per-class metrics
                    for class_name in self.class_names:
                        if class_name in report_dict:
                            mlflow.log_metrics({
                                f"precision_{class_name}": report_dict[class_name]["precision"],
                                f"recall_{class_name}": report_dict[class_name]["recall"],
                                f"f1_{class_name}": report_dict[class_name]["f1-score"],
                            }, step=0)
                    
                    # Log artifacts
                    mlflow.log_artifact(self.config.report, artifact_path="reports")
                    
                    # Log model
                    mlflow.pytorch.log_model(self.model, "evaluated_model")

                    logger.info(f"MLflow run ID: {run.info.run_id}")
                    
            except Exception as e:
                logger.error(f"MLflow logging failed: {e}")
                # Continue even if MLflow fails

        logger.info("Evaluation Completed.")
        return report_dict, cm