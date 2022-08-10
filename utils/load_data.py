""" data loading and processing

Key point: Transformation of dimensions to meet TorchIO request
    
"""
from cmath import e
import os
import torchio as tio
from glob import glob
import natsort
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch
from utils.labels_process import labels_process

"""read data and unity format

    Args:
        Subclass of Dataset

    Returns:
        torchio Subject (image and mask)
    """
class CoreDataset(Dataset):
    
    def __init__(self, transform, train_settings):
        # read the path of images and masks
        self.train_settings = train_settings
        image_folder = self.train_settings["image_folder"]
        mask_folder = self.train_settings["mask_folder"]
        
        self.path_images = natsort.natsorted(glob(os.path.join(image_folder, '*.png')))
        self.path_masks = natsort.natsorted(glob(os.path.join(mask_folder, '*.png')))
        self.transform = transform

        self.labels_process = labels_process(train_settings)
        
    def __len__(self):
        return len(self.path_images)
    
    def __getitem__(self, idx):

        # opencv read image to BGR
        image_read = Image.open(self.path_images[idx])
        mask_read = Image.open(self.path_masks[idx])
        if image_read.mode == 'L':
            pass
        else:
            # L = R * 0.299 + G * 0.587 + B * 0.114
            image_read = image_read.convert('L')

        resize_image = transforms.Resize([480, 610])
        image = resize_image(image_read)

        # change image to array (x,y)
        image = np.array(image)
        # change mask (RGB) to mask (x,y) including 0,1,2,3,...
        mask = mask_read.convert('RGB')
        mask = self.labels_process.rgb2mask(np.array(mask))
        mask = Image.fromarray(mask)
        resize_mask = transforms.Resize([480,610])
        mask = resize_mask(mask)
        mask = np.array(mask)

        # change image (x,y) to tensor image (1,x,y) and change mask (0,1,2,3,...) to tensor
        tensor_image = torch.from_numpy(image).unsqueeze(2).permute(2,0,1).float()
        tensor_mask = torch.from_numpy(mask).long()

        # change image (1,x,y) to image (1,x,y,1) for fitting torchio
        tio_image_4d = torch.unsqueeze(tensor_image,3)

        # change mask (x,y) to mask (1,x,y,1) for fitting torchio
        tio_mask_3d = torch.unsqueeze(tensor_mask,0)
        tio_mask_4d = torch.unsqueeze(tio_mask_3d,3)

        # form class Subject in torchio to do transform 
        data_subject = tio.Subject(
                image=tio.ScalarImage(tensor=tio_image_4d),
                mask=tio.LabelMap(tensor=tio_mask_4d)
            )

        #transform image and mask 
        if self.train_settings["augmentation"] == "True":
            data_mata = [data_subject]
            data_subject = tio.SubjectsDataset(data_mata, transform=self.transform)
            data_subject = data_subject[0]

        return data_subject
    
"""split dataset 

Args:
    val_ratio: ratio of validate to train
    params: the parameters of pytorch.dataloader
    transforms: the types of image transform
    train_settings: path of images and masks

Returns:
    training dataset and validation dataset"""

def make_dataloaders(val_ratio, params, transforms, train_settings):

    # read data and unity format
    dataset = CoreDataset(transforms, train_settings)

    # split the train and validate
    val_len = int(val_ratio*len(dataset))
    lengths = [len(dataset)-val_len, val_len]
    train_dataset, val_dataset = random_split(dataset, lengths, generator=torch.Generator().manual_seed(0))
    

    # DataLoader (read in batches)
    train_loader = DataLoader(train_dataset, drop_last=True, **params)
    val_loader = DataLoader(val_dataset, drop_last=True, **params)
    
    return train_loader, val_loader