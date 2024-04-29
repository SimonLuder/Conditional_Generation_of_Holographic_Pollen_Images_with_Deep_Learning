import os
import sys
import argparse
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

def test(config_file):

    config = load_config(config_file)

    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    inference_config = config["vqvae_inference_params"]

    # create checkpoint paths
    Path(os.path.join(train_config['task_name'], 
                      train_config['vqvae_autoencoder_ckpt_name']
                      )).mkdir(parents=True, exist_ok=True)

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
    dataset = HolographyImageFolder(root=dataset_config["img_path"], transform=transforms)

    # dataloader
    dataloader = DataLoader(dataset,
                            batch_size=train_config['autoencoder_batch_size'],
                            shuffle=False)
    
    # load pretrained vqvae
    model = VQVAE(img_channels=dataset_config['img_channels'], config=autoencoder_config).to(device)
    
    model.load_state_dict(
        torch.load(os.path.join(train_config['task_name'], 
                                train_config['vqvae_autoencoder_ckpt_name'], 
                                inference_config["ckpt"]), 
                                map_location=device))
    model.eval()

    # load pretrained discriminator
    discriminator = PatchGanDiscriminator(img_channels=dataset_config['img_channels']).to(device)

    discriminator.load_state_dict(
        torch.load(os.path.join(train_config['task_name'], 
                                train_config['vqvae_discriminator_ckpt'], 
                                inference_config[["ckpt"]]), 
                                map_location=device))
    discriminator.eval()

    # mse reconstruction criterion
    mse_loss = torch.nn.MSELoss()

    # lpips perceptual criterion
    lpips_model = lpips.LPIPS(net='alex').to(device)

    # dscriminator reconstruction citerion
    discriminator_loss = torch.nn.MSELoss()

    reconstruction_losses = []          # reconstruction loss (l2)
    codebook_losses = []                # codebook loss between predicted and nearest codebook vector
    lpips_losses = []                   # preceptual loss (lpips)
    g_losses = []                       # weighted sum of reconstruction, preceptual and discriminator scores


    with torch.no_grad():
        
        pbar = tqdm(dataloader)
        for (im, _, _) in pbar:
    
            im = im.float().to(device)

            # autoencoder forward pass
            output, z, quantize_losses = model(im)

            # reconstruction loss
            rec_loss = mse_loss(output, im)
            reconstruction_losses.append(rec_loss.item())

            # codebook & commitment loss loss
            g_loss = (rec_loss + 
                      train_config['codebook_weight'] * quantize_losses["codebook_loss"] + 
                      train_config['commitment_beta'] * quantize_losses["commitment_loss"]
                      )
            codebook_losses.append(train_config['codebook_weight'] * quantize_losses['codebook_loss'].item())

            # lpips loss
            lpips_loss = train_config['perceptual_weight'] * torch.mean(lpips_model(output, im))
            lpips_losses.append(train_config['perceptual_weight'] * lpips_loss.item())
            g_loss += lpips_loss

            # discriminator loss (used only from "discriminator_start_step" onwards)
            disc_fake_pred = discriminator(output)
            disc_fake_loss = discriminator_loss(disc_fake_pred,
                                                torch.ones(disc_fake_pred.shape,
                                                           device=disc_fake_pred.device))
            g_loss += train_config['discriminator_weight'] * disc_fake_loss

            g_losses.append(g_loss.item())

            ##############################################################################

        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for vqvae inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/vae_config.yaml', type=str)
    args = parser.parse_args()

    test(args.config_path)


