import torch
import torch.optim as optim
from torchvision import datasets

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations import functional as F

import yaml
from yaml.loader import SafeLoader

SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

class Cifar10SearchDataset(datasets.CIFAR10):
    def __init__(self, root="~/data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

with open("modular/params.yaml") as f:
    params = yaml.load(f, Loader=SafeLoader)


def load_data(mode, transform):
   return Cifar10SearchDataset(root='./data', train=(mode=="train"),
                                        download=True, transform=transform)

def get_transforms(mode):
    means = params["transform_means"]
    stds = params["transform_stds"]

    #REMOVE THIS
    if mode == 'train':
        return A.Compose([
                            A.Normalize(mean=means, std=stds, always_apply=True),
                            A.PadIfNeeded(min_height=36, min_width=36, border_mode=F.cv2.BORDER_REFLECT),
                            A.RandomCrop(height=32, width=32),
                            A.HorizontalFlip(p=0.5),
                            A.CoarseDropout(max_holes=1, max_height=8, max_width=8, p=0.5, fill_value=means),
                            #A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=means),
                            ToTensorV2(),
                        ])
    else:
        return A.Compose([
                            A.Normalize(mean=means, std=stds, always_apply=True),
                            ToTensorV2(),
                        ])

def construct_loader(data):
    dataloader_args = dict(shuffle=True, batch_size=512, num_workers=0, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    return torch.utils.data.DataLoader(data, **dataloader_args)