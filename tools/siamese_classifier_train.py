import os 
import sys
import shutil
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from sklearn.utils.class_weight import compute_class_weight

# add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)

from utils.config import load_config
from utils.wandb import WandbManager
from dataset import PairwiseHolographyImageFolder

class SiameseResNet(nn.Module):
    def __init__(self, base_model, num_out):
        super(SiameseResNet, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(base_model.fc.out_features * 2, num_out)  # Adjust the input size for the combined features

    def forward(self, input1, input2):
        feat1 = self.base_model(input1)
        feat2 = self.base_model(input2)
        combined_features = torch.cat((feat1, feat2), dim=1)
        output = self.fc(combined_features)
        return output

def train(config_path):
    
    # Train on GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the configuration for training
    config = load_config(config_path)
    model_config = config["model"]
    train_config = config["training"]
    dataset_config = config["dataset_params"]

    learning_rate   = train_config["learning_rate"]
    num_epochs      = train_config["epochs"]
    batch_size      = train_config["batch_size"]
    validation_step = train_config["resnet_val_steps"]

    # create checkpoint directory
    Path(os.path.join(train_config["task_name"], train_config["resnet_ckpt_name"])).mkdir(parents=True, exist_ok=True)

    # copy config to checkpoint folder
    shutil.copyfile(config_path, os.path.join(train_config['task_name'], train_config["resnet_ckpt_name"], os.path.basename(config_path)))

    # setup WandbManager
    wandb_manager = WandbManager(project="MSE_P8", run_name=train_config['resnet_ckpt_name'], config=config)
    # init run
    wandb_run = wandb_manager.get_run()

    # Dataset transformations
    transforms_list = []

    transforms_list.append(transforms.ToTensor())

    if dataset_config.get("img_interpolation"):

        transforms_list.append(transforms.Resize((dataset_config["img_interpolation"], 
                                                  dataset_config["img_interpolation"]),
                                                  interpolation = transforms.InterpolationMode.BILINEAR))
        
    transforms_list.append(transforms.Normalize((0.5) * dataset_config["img_channels"], 
                                                (0.5) * dataset_config["img_channels"]))

    transforms_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))  # Convert 1 channel to 3 channels
    
    transform = transforms.Compose(transforms_list)

    # Datasets
    train_dataset = PairwiseHolographyImageFolder(root=dataset_config["img_path"], 
                                          transform=transform, 
                                          config=dataset_config,
                                          labels=dataset_config.get("labels_train"))
    
    val_dataset = PairwiseHolographyImageFolder(root=dataset_config["img_path"], 
                                        transform=transform, 
                                        config=dataset_config,
                                        labels=dataset_config.get("labels_val"))

    # Dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # # Load pre-trained ResNet model
    # model = models.resnet18(pretrained=True)
    
    # # Modify the final layer to fit the number of classes
    # num_ftrs = model.fc.in_features
    # num_out = model_config["out_classes"]
    # model.fc = nn.Linear(num_ftrs, num_out)

    # Load pre-trained ResNet model and modify it for Siamese network
    base_model = models.resnet18(pretrained=True)
    num_out = model_config["out_classes"]
    model = SiameseResNet(base_model, num_out)
    model = model.to(device)


    # Calculate class weights
    all_labels = list(pd.read_csv(dataset_config["labels_train"])[dataset_config["classes"]])
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # set seeds
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    step_count = 1
    
    print("start_training")

    # Training loop
    for epoch in range(num_epochs):

        model.train()

        for inputs, labels, _ in train_loader:
            
            inputs1, inputs2 = inputs  # first image, second image
            labels, _ = labels["class"]

            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs1, inputs2)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Log metrics
            logs = {"epoch": epoch, "step": step_count, "loss": loss.item() / batch_size}


            # Validation
            if step_count % validation_step == 0:

                val_loss = 0.0
                correct = 0
                total = 0
                class_correct = [0] * num_out
                class_total = [0] * num_out

                all_labels = []
                all_preds = []

                with torch.no_grad():

                    # Validation loop
                    for val_inputs, val_labels, _ in val_loader:
                        
                        val_inputs1, val_inputs2 = val_inputs  # first image, second image
                        val_labels, _ = val_labels["class"]

                        val_inputs1, val_inputs2, val_labels = val_inputs1.to(device), val_inputs2.to(device), val_labels.to(device)
                        outputs = model(val_inputs1, val_inputs2)
                        loss = criterion(outputs, val_labels)
                        val_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        total += val_labels.size(0)
                        correct += (predicted == val_labels).sum().item()

                        all_labels.extend(val_labels.cpu().numpy())
                        all_preds.extend(predicted.cpu().numpy())

                        # Calculate per-class metrics
                        for i in range(len(val_labels)):
                            label = val_labels[i].item()
                            class_total[label] += 1
                            if predicted[i] == val_labels[i]:
                                class_correct[label] += 1

                    val_loss /= len(val_loader)

                    logs["val_correct"] = correct
                    logs["val_total"] = total
                    logs["val_accuracy"] = 100 * correct / total
                    logs["val_loss"] = val_loss

                    # Log per-class accuracy
                    for i in range(num_out):
                        if class_total[i] > 0:
                            class_accuracy = 100 * class_correct[i] / class_total[i]
                        else:
                            class_accuracy = 0  # Avoid division by zero
                        logs[f"class_{i}_accuracy"] = class_accuracy

                # Save the labels and predictions to npy files
                output_dir = os.path.join(train_config["task_name"], train_config['resnet_ckpt_name'], "validation", str(step_count))
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                np.save(os.path.join(output_dir, 'labels.npy'), np.array(all_labels))
                np.save(os.path.join(output_dir, 'predictions.npy'), np.array(all_preds))

            # log loss to wandb
            wandb_run.log(data=logs)

            step_count += 1

            # Save the ResNet model weights at defined intervalls
            if step_count % train_config["resnet_ckpt_steps"] == 0:

                torch.save(model.state_dict(), 
                           os.path.join(train_config["task_name"],
                                        train_config['resnet_ckpt_name'],
                                        "latest.pth"))
                torch.save(model.state_dict(), 
                            os.path.join(train_config["task_name"],
                                         train_config['resnet_ckpt_name'],
                                         f"{step_count}.pth"))
            
            
    # Save the ResNet model weights at the end of the training
    torch.save(model.state_dict(), 
                           os.path.join(train_config["task_name"],
                                        train_config['resnet_ckpt_name'],
                                        "latest.pth"))
    torch.save(model.state_dict(), 
                os.path.join(train_config["task_name"],
                                train_config['resnet_ckpt_name'],
                                f"{step_count}.pth"))





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for resnet training')
    parser.add_argument('--config', dest='config_path',
                        default='config/siamese_config.yaml', type=str)
    args = parser.parse_args()

    train(args.config_path)