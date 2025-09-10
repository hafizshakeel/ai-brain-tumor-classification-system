import torch
import torch.nn as nn
import timm
from src.brain_tumor_classification import logger
from src.brain_tumor_classification.entity.config_entity import PrepareBaseModelConfig


# Base Model Preparation
class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None

    def load_model(self):
        logger.info(f"Loading pretrained model: {self.config.model_name}")
        self.model = timm.create_model(
            self.config.model_name,
            pretrained=True,
            num_classes=self.config.num_classes
        )
        logger.info(f"Model {self.config.model_name} loaded successfully.")

        # Save raw base model
        torch.save(self.model, self.config.base_model_path)
        logger.info(f"Base model saved at: {self.config.base_model_path}")

    def freeze_base_layers(self):
        if self.config.freeze_base and self.model is not None:
            logger.info("Freezing base layers...")
            for param in self.model.parameters():
                param.requires_grad = False
            # Keep classifier head trainable
            for param in self.model.get_classifier().parameters():
                param.requires_grad = True
            logger.info("Base layers frozen, classifier head trainable.")

    def save_updated_model(self):
        torch.save(self.model, self.config.updated_model_path)
        logger.info(f"Updated model saved at: {self.config.updated_model_path}")

