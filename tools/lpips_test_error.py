"""
This script calculates the pairwise similarity between images using the Learned 
Perceptual Image Patch Similarity (LPIPS) metric and Mean Squared Error (MSE). 
"""


import os
import sys
import pickle
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torch.nn.functional import mse_loss

# add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)

from dataset import HolographyImageFolder
from utils.config import load_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_pairwise_similarity(config_file):

    config = load_config(config_file)

    dataset1_config = config['dataset1_params']
    dataset2_config = config['dataset2_params']
    test_config = config["test_params"]

    # Create checkpoint paths
    Path("lpips").mkdir(parents=True, exist_ok=True)

    # Transforms
    transforms_list = [torchvision.transforms.ToTensor()]

    if dataset1_config.get("img_interpolation"):
        transforms_list.append(torchvision.transforms.Resize((dataset1_config["img_interpolation"], 
                                                              dataset1_config["img_interpolation"]),
                                                              interpolation=torchvision.transforms.InterpolationMode.BILINEAR))

    transforms_list.append(torchvision.transforms.Normalize((0.5) * dataset1_config["img_channels"], 
                                                            (0.5) * dataset1_config["img_channels"]))

    transforms = torchvision.transforms.Compose(transforms_list)

    # Datasets
    dataset1 = HolographyImageFolder(root=dataset1_config['img_path'], 
                                     transform=transforms, 
                                     config=dataset1_config, 
                                     labels=dataset1_config['labels_test'])
    
    dataset2 = HolographyImageFolder(root=dataset2_config['img_path'], 
                                     transform=transforms, 
                                     config=dataset2_config, 
                                     labels=dataset2_config['labels_test'])

    # Dataloaders
    dataloader1 = DataLoader(dataset1, batch_size=1, shuffle=False)
    dataloader2 = DataLoader(dataset2, batch_size=1, shuffle=False)

    # LPIPS perceptual criterion
    lpips_model = LPIPS(net_type='alex').to(device)

    lpips_scores = []
    mse_scores = []
    class_ids = []
    filenames = []

    with torch.no_grad():
        for (im1, labels, filename), (im2, _, _) in tqdm(zip(dataloader1, dataloader2), total=len(dataloader1)):
            im1 = im1.float().to(device)
            im2 = im2.float().to(device)

            im1_lpips = torch.clamp(im1, -1., 1.)
            im2_lpips = torch.clamp(im2, -1., 1.)

            if im1_lpips.shape[1] == 1:
                im1_lpips = im1_lpips.repeat(1, 3, 1, 1)
                im2_lpips = im2_lpips.repeat(1, 3, 1, 1)

            # Compute LPIPS score
            lpips_score = lpips_model(im1_lpips, im2_lpips).item()
            lpips_scores.append(lpips_score)

            # Compute MSE score
            mse_score = mse_loss(im1, im2).item()
            mse_scores.append(mse_score)

            class_ids.append(labels["class"].item())
            filenames.append(filename[0])

    # Save results
    with open(test_config['output_dir'], "wb") as f:
        pickle.dump({
            "lpips_scores": lpips_scores, 
            "mse_scores": mse_scores, 
            "class_ids": class_ids, 
            "rec_path": filenames
        }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for LPIPS pairwise similarity calculation')
    parser.add_argument('--config', dest='config_path', default='config/lpips_test_config.yaml', type=str)

    args = parser.parse_args()

    calculate_pairwise_similarity(args.config_path)
