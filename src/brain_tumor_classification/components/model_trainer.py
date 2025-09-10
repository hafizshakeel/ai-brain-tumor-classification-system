
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import mlflow
import os
import shutil

from src.brain_tumor_classification.entity.config_entity import TrainingConfig
from src.brain_tumor_classification import logger



class TrainingPipeline:
    def __init__(self, config: TrainingConfig, model: torch.nn.Module, device, mlflow_config):
        self.config = config
        self.model = model
        self.device = device

        self.mlflow_config = mlflow_config

        # Setup MLflow
        if self.mlflow_config.log_with_mlflow:
            mlflow.set_tracking_uri(self.mlflow_config.mlflow_tracking_uri)
            mlflow.set_experiment(self.mlflow_config.mlflow_experiment)

        # Transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # Dataset + Split
        full_dataset = datasets.ImageFolder(self.config.train_data_dir, transform=self.train_transform)
        val_size = int(len(full_dataset) * self.config.val_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        self.val_dataset.dataset.transform = self.val_transform

        # Dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config.batch_size, shuffle=False)

        # Loss, optimizer, scheduler
        self.criterion = nn.CrossEntropyLoss()
        if config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        if config.scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config.step_size, gamma=config.gamma)
        elif config.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        else:
            self.scheduler = None

        # Save class names (for reports)
        self.class_names = full_dataset.classes

    def train(self, epoch):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        if self.scheduler:
            self.scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        # Log to MLflow
        if self.mlflow_config.log_with_mlflow:
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_acc, step=epoch)

        logger.info(f"Epoch [{epoch}] Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        return epoch_loss, epoch_acc

    def validate(self, epoch):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        # Log to MLflow
        if self.mlflow_config.log_with_mlflow:
            mlflow.log_metric("val_loss", epoch_loss, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_acc, step=epoch)

        logger.info(f"Epoch [{epoch}] Val Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

        # Metrics (per class)
        report = classification_report(all_labels, all_preds, target_names=self.class_names, digits=4, output_dict=True, zero_division=0)
        logger.info(f"\nClassification Report (Epoch {epoch}):\n{report}")

        # Log per-class metrics to MLflow
        if self.mlflow_config.log_with_mlflow:
            for class_name in self.class_names:
                if class_name in report:
                    mlflow.log_metric(f"precision_{class_name}", report[class_name]['precision'], step=epoch)
                    mlflow.log_metric(f"recall_{class_name}", report[class_name]['recall'], step=epoch)
                    mlflow.log_metric(f"f1_{class_name}", report[class_name]['f1-score'], step=epoch)

        return epoch_loss, epoch_acc, report


    def run(self):
        logger.info("Starting Training...")
        best_acc = 0.0
        
        # Start MLflow run
        if self.mlflow_config.log_with_mlflow:
            mlflow.start_run()
            # Log parameters
            mlflow.log_params({
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "optimizer": self.config.optimizer,
                "scheduler": self.config.scheduler,
                "val_split": self.config.val_split,
                "model_name": "swin_tiny_patch4_window7_224"
            })

        try:
            for epoch in range(1, self.config.epochs + 1):
                train_loss, train_acc = self.train(epoch)
                val_loss, val_acc, report = self.validate(epoch)

                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(self.model, self.config.trained_model_path)
                    logger.info(f"Best model saved at {self.config.trained_model_path} with Val Acc: {best_acc:.2f}%")
                    
                    # copy the best model to the weights folder in main directory if don't want to push artifacts folder to github


                    # Log best model to MLflow as an artifact
                    if self.mlflow_config.log_with_mlflow:
                        mlflow.log_artifact(self.config.trained_model_path, artifact_path="models")
                        mlflow.log_metric("best_val_accuracy", best_acc, step=epoch)

            logger.info("Training Completed.")
            
            # Log the final model as an artifact
            if self.mlflow_config.log_with_mlflow:
                final_model_path = os.path.join(self.config.root_dir, "final_model.pth")
                torch.save(self.model, final_model_path)
                mlflow.log_artifact(final_model_path, artifact_path="models")
                mlflow.log_metric("final_val_accuracy", val_acc)
                
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
        
        finally:
            if self.mlflow_config.log_with_mlflow:
                mlflow.end_run()

