import os
import torch
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
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
    
    def __init__(self, root, transform=None, labels=None, config=None, conditioning=None):
        self.root = root
        self.transform = transform
        self.config = config
        self.labels = labels
        self.cond_imgs = None
        self.tabular_features = None
        self.class_labels = None

        if labels.endswith(".csv") and os.path.exists(labels):
            print("Loading dataset from csv")
            self.load_csv_dataset()
        
        elif labels.endswith(".pkl") and os.path.exists(labels):
            print("Loading dataset from pickle")
            self.load_pickle_dataset()

        else:
            print("Seach everything. This might take some time.")
            load_pickle = False
            save_pickle = False
            seach_root_dir = False

            if self.labels is None:
                seach_root_dir = True
            else:
                if not os.path.exists(self.labels):
                    seach_root_dir = True
                    save_pickle = True
                else:
                    load_pickle = True

            foldername_to_id = dict()
            index = 0
            if seach_root_dir:
                # get all images in root and subdirs of root
                self.samples = []
                for dirpath, dirnames, filenames in os.walk(self.root):
                    for filename in filenames:
                        if filename.endswith(('.png', '.jpg', '.jpeg')):  # add more extensions as needed
                            img_path = os.path.join(dirpath, filename)
                            folder_name = os.path.basename(os.path.dirname(img_path))
                            if folder_name not in foldername_to_id.keys():
                                foldername_to_id[folder_name] = index
                                index += 1
                            self.samples.append((img_path, foldername_to_id[folder_name], filename))

                if save_pickle:
                    # save samples as pickle
                    Path(os.path.dirname(os.path.split(self.labels)[-1])).mkdir(parents=True, exist_ok=True)
                    with open(self.labels, 'wb') as f:
                        pickle.dump(self.samples, f)
            
            if load_pickle:
                # load samples from pickle file if exists
                with open(self.labels, 'rb') as f:
                    self.samples = pickle.load(f)
   
    def load_csv_dataset(self):
        
        df = pd.read_csv(self.labels)
        class_cond_colunmn = self.config.get("classes", None)
        feature_columns = self.config.get("features", None)
        cond_image_colunmn = self.config.get("cond_img_path", None)
        filename_column = self.config["filenames"]
        image_folder = self.config["img_path"]

        filenames = [filename for filename in list(df[filename_column])]
        img_paths = [os.path.join(image_folder, filename) for filename in filenames]
        self.samples = list(zip(img_paths, filenames))

        # class features for conditioning
        if class_cond_colunmn is not None:
            self.class_labels = df[class_cond_colunmn].values.tolist()
        else:
            self.class_labels = None

        # tabular features for conditioning
        if feature_columns is not None:
            self.tabular_features = df[feature_columns].values.tolist()
        else:
            self.tabular_features = None
            
        # images for conditioning
        if cond_image_colunmn is not None:
            self.cond_imgs = df[cond_image_colunmn].values.tolist()
        else:
            self.cond_imgs = None
        
    def load_pickle_dataset(self):
        # load samples from pickle file if exists
        with open(self.labels, 'rb') as f:
            self.samples = pickle.load(f)
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        # get image
        img_path, filename = self.samples[idx]
        img = Image.open(img_path)

        if img.mode == 'I':
            img = img.convert('I;16')
            
        img = (np.array(img) / 256).astype('uint8')

        if self.transform:
            img = self.transform(img)   

        # get conditioning
        condition = dict()

        if self.class_labels:
            condition["class"] = self.class_labels[idx]

        if self.tabular_features:
            tab_cond = self.tabular_features[idx]
            condition["tabular"] = torch.tensor(tab_cond)

        if self.cond_imgs:
            cond_image = self.cond_imgs[idx]
            # TODO Open image as tensor and preprocess
            # cond_image = torch.flatten(cond_image, start_dim=1)
            # condition = torch.cat([condition, cond_image], dim=-1)
            condition["image"] = cond_image

        return img, condition, filename




