import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import yaml
import json


# Load params.yaml for consistency
def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def load_class_names(train_data_dir):
    """Get class names from the training directory structure"""
    return sorted(os.listdir(train_data_dir))


class Predictor:
    def __init__(self, model_path: str, train_data_dir: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # load params
        params = load_params()
        image_size = params["base_model"]["image_size"]

        # preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # load model
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.eval()

        # get class names
        self.class_names = load_class_names(train_data_dir)

    def predict(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        CLASS_LABELS = {
            "glioma_tumor": "Glioma Tumor",
            "meningioma_tumor": "Meningioma Tumor",
            "pituitary_tumor": "Pituitary Tumor",
            "no_tumor": "No Tumor"
        }

        pred_class = self.class_names[pred.item()]
        human_label = CLASS_LABELS.get(pred_class, pred_class)
        
        # Create a dictionary of class probabilities
        prob_dict = {}
        for i, class_name in enumerate(self.class_names):
            prob_dict[CLASS_LABELS.get(class_name, class_name)] = float(probs[0][i].item())

        return {
            "image": image_path,
            "prediction": human_label,
            "confidence": float(conf.item()),
            "probs": prob_dict
        }


if __name__ == "__main__":
    # load config.yaml for paths
    with open("config\config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_path = config["training"]["trained_model_path"]
    train_data_dir = config["training"]["train_data_dir"]

    predictor = Predictor(model_path, train_data_dir)

    # Example prediction
    test_image = "artifacts/data_ingestion/data/Testing/glioma_tumor/image(1).jpg"
    result = predictor.predict(test_image)

    print(json.dumps(result, indent=4))





