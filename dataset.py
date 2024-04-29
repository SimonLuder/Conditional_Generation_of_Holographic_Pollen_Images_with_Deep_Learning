import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets import ImageFolder
from typing import Any, Tuple


class ImageClassDataset(ImageFolder):
    
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, path) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


class ImageSentenceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches of Images and Prompts.
    
    Attributes:
        transform (callable, optional): Optional transform to be applied on a sample.
        image_files (list): List of image file paths.
        captions (list): List of image captions.
    """
    def __init__(self, labels_path, transform=None, preprocess=None):
        self.transform = transform
        self.preprocess = preprocess
        df = pd.read_csv(labels_path)
        self.image_files = df["file"].tolist()
        self.captions = df["prompt"].tolist()
        
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        caption = self.captions[idx]
        image = Image.open(filename)

        if self.preprocess:
            caption = self.preprocess(caption)

        if self.transform:
            image = self.transform(image)

        return image, caption, filename


class ImageTabularDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches of Images and feature vectors.
    The feature vectors have been one-hot encoded for categorical values and column-wise normalized.
    
    Attributes:
        transform (callable, optional): Optional transform to be applied on a sample.
        image_files (list): List of image file paths.
        captions (list): feature vectors.
    """
    def __init__(self, labels_path, transform=None):
        self.transform = transform
        df = pd.read_csv(labels_path)
        self.image_files = df["file"].tolist()
        # captions
        captions = df[["radius", "x", "y", "rotation", "aspect_ratio", "shape_name"]]
        captions = pd.get_dummies(captions, columns = ['shape_name'], dtype=int, drop_first=False)
        captions=(captions - captions.mean()) / captions.std()
        self.captions = captions.values
    
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        caption = torch.Tensor(self.captions[idx])
        image = Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, caption, filename
    

class ImageImageDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches of Images as tensor and Images raw.
    The feature vectors have been one-hot encoded for categorical values and column-wise normalized.
    
    Attributes:
        transform (callable, optional): Optional transform to be applied on a sample.
        image_files (list): List of image file paths.
    """
    def __init__(self, labels_path, transform=None, preprocess=None):
        self.transform_img = transform
        self.transform_cond = preprocess
        df = pd.read_csv(labels_path)
        self.image_files = df["file"].tolist()

       
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image = Image.open(filename)
        condition = image

        if self.transform_cond:
            condition = self.transform_cond(condition)

        if self.transform_img:
            image = self.transform_img(image)
            
        return image, condition, filename
    

class HolographyImageFolder(torch.utils.data.Dataset):
    
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transform = transform

        self.samples = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith(('.png', '.jpg', '.jpeg')):  # add more extensions as needed
                    img_path = os.path.join(dirpath, filename)
                    folder_name = os.path.basename(os.path.dirname(img_path))
                    self.samples.append((img_path, folder_name, filename))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, folder_name, filename = self.samples[idx]
        img = Image.open(img_path)
        if img.mode == 'I':
            img = img.convert('I;16')
        img = (np.array(img) / 256).astype('uint8')
        if self.transform:
            img = self.transform(img)
        return img, folder_name, filename