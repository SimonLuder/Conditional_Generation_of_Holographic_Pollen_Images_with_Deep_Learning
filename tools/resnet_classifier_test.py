import os
import sys
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix

# Add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.config import load_config
from dataset import HolographyImageFolder

def test(config_path):
    try:
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the configuration for testing
        config = load_config(config_path)
        model_config = config["model"]
        dataset_config = config["dataset_params"]
        test_config = config["testing"]

        # create checkpoint directory
        Path(os.path.join(test_config["task_name"], test_config["output_dir"])).mkdir(parents=True, exist_ok=True)

        # Dataset transformations
        transforms_list = [
            transforms.ToTensor(),
            transforms.Resize((dataset_config["img_interpolation"], dataset_config["img_interpolation"]),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize([0.5] * dataset_config["img_channels"], [0.5] * dataset_config["img_channels"]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert 1 channel to 3 channels
        ]
        transform = transforms.Compose(transforms_list)

        # Test dataset
        test_dataset = HolographyImageFolder(root=dataset_config["img_path"], 
                                             transform=transform, 
                                             config=dataset_config,
                                             labels=dataset_config.get("labels_test"))

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                  batch_size=test_config["batch_size"], 
                                                  shuffle=False)

        # Load pre-trained ResNet model
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        num_out = model_config["out_classes"]
        model.fc = nn.Linear(num_ftrs, num_out)
        model = model.to(device)

        # Load the saved model weights
        model.load_state_dict(torch.load(test_config["model_ckpt"]))
        model.eval()

        # Metrics
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                labels = labels["class"]
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Calculate metrics
        print("Classification Report:")
        print(classification_report(all_labels, all_preds))
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))

    except Exception as e:
        print(f"Error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for resnet testing')
    parser.add_argument('--config', dest='config_path', default='config/resnet_config.yaml', type=str)
    args = parser.parse_args()

    test(args.config_path, args.model_weights_path)
