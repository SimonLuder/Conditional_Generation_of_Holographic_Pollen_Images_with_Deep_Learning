# Generative_Diffusion_Models_for_3D_Geometric_Objects

This repository contains the code for the report: Conditional Generation of Holographic Pollen Images with Deep Learning. 
The project was created as part of the P8 in the Masters of Science in Engineering (MSE) study program at FHNW. The aim of this work was to generate holographic pollen images using a latent diffusion model. The generated images are intended to be used to expand the existing database.


![alt text](./images/linear_holo.jpg?raw=true)


## Structure
------------
```
├── README.md                           <- The file you are reading right now
│
├── config                              <- Contains configuration files to run the models (separate download)
│
├── dataset.py                          <- Contains the dataset classes
│
├── dockerfile
│   └── Dockerfile.slurm                <- Dockerfile for training models on slurm     
│         
├── embedding.py                        <- Code for the various embedding classes 
│
├── holographic_pollen                  <- Containes the trained models, logged metrics and results (separate download)
│
├── model              
│   ├── ddpm.py                         <- Contains the DDPM model class
│   ├── blocks.py                       <- Contains the basic bbuilding blocks for the models
│   ├── vqvae.py                        <- Contains the modular VQ-VAE model class
│   ├── unet_v2.py                      <- Contains the reworked modular UNet model class
│   └── discriminator.py                <- Contains the PatchGan discriminator model class
│
├── notebooks  
│   ├── eda_poleno.ipynb                <- Contains code to describe, clean and split the poleno dataset into train, val and test sets.
│   ├── vqvae_test_results.ipynb        <- Contains the logged metrics from validating and testing the VQ-VAE models
│   └── ldm_test_results.ipynb          <- Contains the results from the evaluation of the Latent diffusion models
│
├── tools       
│   ├── recalculate_poleno_features.py  <- Recalculates the visual features for the poleno dataset
│   ├── vqvae_train.py                  <- To train the VQ-VAE 
│   ├── vqvae_validate.py               <- Validation loop for the VQ-VAE
│   ├── vqvae_test.py                   <- Test the VQ-VAE
│   ├── ldm_train.py                    <- Train the LDM 
│   ├── ldm_validate.py                 <- Validation loop for the LDM
│   ├── ldm_test.py                     <- To test the LDM   
│   ├── siamese_classifier_train.py     <- To train and validate the siamese classifier     
│   ├── siamese_classifier_test.py      <- To test and the siamese classifier 
│   └── lpips_test_error.py             <- Recalculates the LPIPS and MSE on the generated images
│
├── utils                    
│   ├── train_test_utils.py             <- Contains general utils for training and testing
│   ├── data_processing.py              <- Methods to handle the Poleno dataset
│   └── wandb.py                        <- Contains WandB functionalities for logging       
│
├── requirements.txt                    <- The requirements file
│
├── images                              <- Contains images created for the report
│
└── runs                                <- This folder contains the individual trained models and logs during training
```
------------


## Docker
To run code inside the dockerfile

Build the holo_images docker image from Dockerfile:
``` sh
docker build -f Dockerfile.slurm -t holo_images .
```

Start the holo_images container in bash
``` sh
docker run -it --rm -v .:/app/ --gpus all holo_images bash
```

Transform the docker image to .tar file for training on slurm
``` sh
docker save holo_images > holo_images.tar

```

------------
## Recalculation of the visual features

The visual features from the holographic images can be recalculated using the subsequent command.

``` sh
python recalculate_poleno_features.py --database YOUR_PATH/Poleno/poleno_marvel.db --image_folder YOUR_PATH/Poleno/
```

## VQ-VAE training and evaluation

VQ-VAE model training and testing can be started using the following commands. The model and training can be customized via the `vqvae_config.yaml` file.


``` sh
# training
python vqvae_train.py --config config/vqvae_config.yaml

# testing
python vqvae_test.py --config config/vqvae_config.yaml
```

## Latent Diffusion Model training and evaluation


The Latent Diffusion models can be trained and tested by the following commands. The model and training can be adapted in the the `ldm_config.yaml` file.


``` sh
# training
python ldm_train.py --config config/ldm_config.yaml

# testing
python ldm_test.py --config config/ldm_config.yaml
```


## Siamese Classifier training and evaluation

The following commands can be used to train and test the siamese classifier. The training configuration can be modified in the `siamese_classifier_config.yaml` file.

``` sh
# training
python siamese_classifier_train.py --config config/siamese_config.yaml

# testing
python siamese_classifier_test.py --config config/siamese_config.yaml
```