import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision

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


def train(config_path):

    config = load_config(config_path)
    dataset_config = config['dataset_params']
    ddpm_config = config['ddpm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup WandbManager
    wandb_manager = WandbManager(project="MSE_P8", run_name=train_config['task_name'] + "_ldm", config=config)
    # init run
    wandb_run = wandb_manager.get_run()

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
    dataset = HolographyImageFolder(root=dataset_config["img_path"], transform=transforms)

    # dataloader
    dataloader = DataLoader(dataset,
                            batch_size=train_config['ldm_batch_size'],
                            shuffle=True)
    
    ################################## autoencoder ###################################
    # Load Autoencoder if latents are not precalculated or are missing
    if os.path.exists(train_config["vqvae_latents_representations"]):
        latents_available = len(os.listdir(train_config["vqvae_latents_representations"])) > 0
    else:
        latents_available = False

    if not latents_available:

        print('Loading vqvae model as no latents found')
        vae = VQVAE(img_channels=dataset_config['img_channels'], config=autoencoder_model_config).to(device)
        vae.eval()

        # Load vae if found
        vae_ckpt_path = os.path.join(train_config['task_name'],
                                     train_config['vqvae_autoencoder_ckpt_name'],
                                     "latest.pth"
                                     )
        
        if os.path.exists(vae_ckpt_path):

            vae.load_state_dict(torch.load(vae_ckpt_path, map_location=device))
            print('Loaded autoencoder checkpoint')

        
        for param in vae.parameters():
            param.requires_grad = False
    ##################################################################################

    # Load UNet
    model = UNet(img_channels=autoencoder_model_config['z_channels'],
                 model_config=ddpm_config).to(device)
    model.train()
    
    # Load Diffusion Process
    input_dummy = torch.zeros(next(iter(dataset))[0].shape).unsqueeze(0).to(device)
    output_dummy, _ = vae.encode(input_dummy)
    ddpm_input_size = output_dummy.shape[-1]
    
    diffusion = DDPMDiffusion(img_size=ddpm_input_size, 
                              img_channels=autoencoder_model_config['z_channels'],
                              noise_schedule="linear", 
                              beta_start=train_config["ldm_beta_start"], 
                              beta_end=train_config["ldm_beta_end"],
                              device=device,
                              )

    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()
    step_count = 0

    for epoch_idx in range(num_epochs):
        losses = []

        pbar = tqdm(dataloader)
        for (im, _, _) in pbar:

            optimizer.zero_grad()
    
            im = im.float().to(device)

            # autencode samples
            if not latents_available:
                with torch.no_grad():
                    im, _ = vae.encode(im)

            # sample timestep
            t = diffusion.sample_timesteps(im.shape[0]).to(device)

            # noise image
            x_t, noise = diffusion.noise_images(im, t)

            # predict noise
            noise_pred = model(x_t, t)
            
            loss = criterion(noise_pred, noise)

            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            pbar.set_postfix(Loss=loss)
            step_count += 1

            logs = {"epoch" :               epoch_idx + 1,
                    "step" :                step_count + 1,
                    "loss" :                np.mean(losses),
                    }
            
            # wandb logging
            wandb_run.log(data=logs)
        
        ################################ model saving ################################
        if step_count % train_config["ldm_ckpt_steps"] == 0:

            torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                        train_config['ldm_ckpt_name'],
                                                        "latest.pth"))
                    
            torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                        train_config['ldm_ckpt_name'],
                                                        f"{step_count}.pth"))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/vae_config.yaml', type=str)
    args = parser.parse_args()

    train(args.config_path)