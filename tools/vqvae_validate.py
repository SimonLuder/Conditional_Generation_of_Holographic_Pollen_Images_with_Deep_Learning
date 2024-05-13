import os
import sys
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import lpips

import torch
from torch.utils.data import DataLoader
import torchvision

# add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)

from model.vqvae import VQVAE
from model.discriminator import PatchGanDiscriminator
from dataset import HolographyImageFolder
from utils.config import load_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate(model, discriminator, config_file):

    # global model
    global lpips_model
    global dataloader_val

    config = load_config(config_file)

    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    inference_config = config["vqvae_inference_params"]

    # create checkpoint paths
    Path(os.path.join(train_config['task_name'], 
                      train_config['vqvae_autoencoder_ckpt_name']
                      )).mkdir(parents=True, exist_ok=True)

    if not "dataloader_val" in globals(): # singleton design pattern
        # transforms
        transforms_list = []

        transforms_list.append(torchvision.transforms.ToTensor())

        if dataset_config.get("img_interpolation"):
            transforms_list.append(torchvision.transforms.Resize((dataset_config["img_interpolation"], 
                                                                dataset_config["img_interpolation"]),
                                                                interpolation = torchvision.transforms.InterpolationMode.BILINEAR))

        transforms_list.append(torchvision.transforms.Normalize((0.5) * dataset_config["img_channels"], 
                                                                (0.5) * dataset_config["img_channels"]))

        transforms = torchvision.transforms.Compose(transforms_list)

        #dataset
        dataset_val = HolographyImageFolder(root=dataset_config["img_path"], 
                                        transform=transforms, 
                                        pkl_path=dataset_config.get("pkl_path_val"))

        # dataloader
        dataloader_val = DataLoader(dataset_val,
                                batch_size=train_config['autoencoder_batch_size'],
                                shuffle=False)
        
        print("Instanciate validation dataloader")

    # load pretrained vqvae
    # if not "model" in globals(): # singleton design pattern
    #     print("Instanciate pretrained model for testing")
    #     model = VQVAE(img_channels=dataset_config['img_channels'], config=autoencoder_config).to(device)
    
    #     model.load_state_dict(
    #         torch.load(os.path.join(train_config['task_name'], 
    #                                 train_config['vqvae_autoencoder_ckpt_name'], 
    #                                 inference_config["model_ckpt"]), 
    #                                 map_location=device))
    model.eval()
    discriminator.eval()

    # mse reconstruction criterion
    mse_loss = torch.nn.MSELoss()

    # lpips perceptual criterion
    if not "lpips_model" in globals(): # singleton design pattern
        print("Instanciate lpips for testing")
        lpips_model = lpips.LPIPS(net='alex').to(device)

    folders = []
    sample_filenames = []
    reconstruction_losses = []          # reconstruction loss (l2)
    codebook_losses = []                # codebook loss between predicted and nearest codebook vector
    lpips_losses = []                   # preceptual loss (lpips)
    
    with torch.no_grad():

        pbar = tqdm(dataloader_val)
        for (im, folder, filenames) in pbar:

            im = im.float().to(device)

            folders.append(folder)
            sample_filenames.append(filenames)

            # autoencoder forward pass
            output, z, quantize_losses = model(im)

            # reconstruction loss
            rec_loss = mse_loss(output, im)
            reconstruction_losses.append(rec_loss.item())

            # codebook & commitment loss
            codebook_losses.append(quantize_losses["codebook_loss"].item())

            # lpips loss
            lpips_loss = torch.mean(lpips_model(output, im))
            lpips_losses.append(lpips_loss.item())

    logs = {"val_epoch_reconstructon_loss"    : np.mean(reconstruction_losses),
            "val_epoch_codebook_loss"         : np.mean(codebook_losses),
            "val_epoch_lpips_loss"            : np.mean(lpips_losses)
            }
            
    model.train()
    discriminator.train()
    return logs

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for vqvae inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/ldm_config.yaml', type=str)
    args = parser.parse_args()

    validate(args.config_path)


