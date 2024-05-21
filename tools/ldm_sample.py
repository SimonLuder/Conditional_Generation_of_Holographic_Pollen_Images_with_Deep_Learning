import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.decomposition import PCA

import torch
import torch.nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import make_grid

# add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)

from model.unet_v2 import UNet
from model.vqvae import VQVAE
from model.ddpm import Diffusion as DDPMDiffusion
from utils.wandb import WandbManager
from utils.config import load_config
from dataset import HolographyImageFolder


def inference(config_path):

    config = load_config(config_path)
    dataset_config = config['dataset_params']
    ddpm_model_config = config['ddpm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['ldm_train_params']
    inference_config = config["ddpm_inference_params"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Path(os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'], 'inference')).mkdir(parents=True, exist_ok=True)

    ################################### transforms ###################################
    transforms_list = []

    transforms_list.append(torchvision.transforms.ToTensor())

    if dataset_config.get("img_interpolation"):
        transforms_list.append(torchvision.transforms.Resize((dataset_config["img_interpolation"], 
                                                              dataset_config["img_interpolation"]),
                                                              interpolation = torchvision.transforms.InterpolationMode.BILINEAR))

    transforms_list.append(torchvision.transforms.Normalize((0.5) * dataset_config["img_channels"], 
                                                            (0.5) * dataset_config["img_channels"]))

    transforms = torchvision.transforms.Compose(transforms_list)

    ###################################### data ######################################

    #dataset
    dataset = HolographyImageFolder(root=dataset_config["img_path"], 
                                transform=transforms, 
                                pkl_path=dataset_config["pkl_path_test"])

    # dataloader
    dataloader = DataLoader(dataset,
                            batch_size=train_config['ldm_batch_size'],
                            shuffle=False)
    
    ################################## autoencoder ###################################
    # Load Autoencoder
    vae = VQVAE(img_channels=dataset_config['img_channels'], config=autoencoder_model_config).to(device)
    vae.eval()

    vae_ckpt_path = os.path.join(train_config['task_name'],
                                 train_config['vqvae_ckpt_dir'],
                                 train_config['vqvae_ckpt_model']
                                 )

    vae.load_state_dict(torch.load(vae_ckpt_path, map_location=device))
    print(f'Loaded autoencoder checkpoint from {vae_ckpt_path}')

    for param in vae.parameters():
        param.requires_grad = False

    ##################################### u-net ######################################
    # Load UNet
    model = UNet(img_channels=autoencoder_model_config['z_channels'], model_config=ddpm_model_config).to(device)
    model.eval()

    unet_ckpt_path = os.path.join(train_config['task_name'], 
                                  train_config['ldm_ckpt_name'], 
                                  inference_config["ddpm_model_ckpt"]
                                  )
       
    model.load_state_dict(torch.load(unet_ckpt_path, map_location=device))
    print(f'Loaded unet checkpoint from {unet_ckpt_path}')
    

    ################################### diffusion ####################################
    # init diffusion class
    if dataset_config.get('img_interpolation'):
        img_size = dataset_config['img_interpolation'] // 2 ** sum(autoencoder_model_config['down_sample'])
    else:
        img_size = dataset_config['img_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
 
    diffusion = DDPMDiffusion(img_size=img_size, 
                              img_channels=autoencoder_model_config['z_channels'],
                              noise_schedule="linear", 
                              beta_start=train_config["ldm_beta_start"], 
                              beta_end=train_config["ldm_beta_end"],
                              device=device,
                              )
    
    print(len(dataloader))
 
    # for (_, _, _) in dataloader:
    while True:

        with torch.no_grad():
            img_latent = diffusion.sample(model, 
                                          condition=None, 
                                          n=train_config['ldm_batch_size'], 
                                          cfg_scale=0,
                                          to_uint8=False)

            # upsample with vqvae
            img = vae.decode(img_latent)

            img_latent = torch.clamp(img_latent, -1., 1.)
            img_latent = (img_latent + 1) / 2

            img = torch.clamp(img, -1., 1.)
            img = (img + 1) / 2

            # pca image reduction
            if img_latent.shape[1] > 3:
                img_latent = pca_channel_reduction(img_latent, out_channels=3)

            # interpolate encoded output to same size as input
            if inference_config['upscale_latent_dim']:
                img_latent = F.interpolate(img_latent, size=(img.shape[-2], img.shape[-1]), mode="nearest")

            latent_grid = make_grid(img_latent.cpu(), nrow=inference_config['num_grid_rows'])
            output_grid = make_grid(img.cpu(), nrow=inference_config['num_grid_rows'])

            latent_grid = torchvision.transforms.ToPILImage()(latent_grid)
            output_grid = torchvision.transforms.ToPILImage()(output_grid)

            latent_grid.save(os.path.join(train_config['task_name'], 
                                          train_config['ldm_ckpt_name'], 
                                          'inference', 
                                          'latents_at_t0.png'))
            
            output_grid.save(os.path.join(train_config['task_name'], 
                                          train_config['ldm_ckpt_name'], 
                                          'inference', 
                                          'output_images.png'))
        
        break

def pca_channel_reduction(batch, out_channels = 3):

    # Initialize PCA
    pca = PCA(n_components=out_channels)

    # List to store PCA transformed images
    x_pca_batch = []

    # Loop over each image in the batch
    for im in batch:
        # Get the shape of the image
        C, H, W = im.shape

        # Flatten the image
        x_2d = im.view(C, -1).cpu().numpy()

        # Fit PCA on the image
        pca.fit(x_2d)

        # Transform the image using PCA and reshape it back to original shape
        x_pca = pca.components_.reshape(out_channels, H, W)
        x_pca = ((x_pca - x_pca.min()) / (x_pca.max() - x_pca.min()))

        # Append the PCA transformed image to the list
        x_pca_batch.append(x_pca)

    # Convert the list to a tensor
    x_pca_batch = [torch.from_numpy(arr) for arr in x_pca_batch]
    x_pca_batch = torch.stack(x_pca_batch)

    return x_pca_batch



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Arguments for ldm inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/ldm_config.yaml', type=str)
    args = parser.parse_args()

    inference(args.config_path)