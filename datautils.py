import torch
import torch.nn as nn

import torchvision
from torchvision import datasets

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np

SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

def load_data(mode):
    return datasets.CIFAR10(root='./data', train=(mode=="train"), download=True)

def transform_data(data, transform_object):
    data.transform = lambda img: transform_object(image=np.array(img))["image"]
    return data

def load_transform_object(mode):
    if mode=="train":
        return A.Compose([A.HorizontalFlip(p=0.5), 
                            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5), 
                            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes = 1, min_height=16,                                                  min_width=16,                                                      fill_value=[0.4914, 0.4822, 0.4465], mask_fill_value = None, p=0.5), 
                            A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
                            ToTensorV2() ], 
                            additional_targets={"image": "image"})
    else:
        return A.Compose([A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ToTensorV2()], 
                            additional_targets={"image": "image"})

def construct_loader(data):
    dataloader_args = dict(shuffle=True, batch_size=512, num_workers=0, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    return torch.utils.data.DataLoader(data, **dataloader_args)
    
def load_class_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']