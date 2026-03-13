# utils/dataset_loader.py
import os
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_transforms(img_size: int = 224, augment: str = "light"):
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if augment == "none":
        train_tfms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # conservative augmentations for MRI
        train_tfms = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            normalize,
        ])
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tfms, eval_tfms

def create_dataloaders(
    data_root: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: str = "light",
    shuffle_train: bool = True,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], int]:

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")
    assert os.path.isdir(train_dir), f"Missing: {train_dir}"
    assert os.path.isdir(val_dir), f"Missing: {val_dir}"
    assert os.path.isdir(test_dir), f"Missing: {test_dir}"

    train_tfms, eval_tfms = get_transforms(img_size, augment)

    train_ds = datasets.ImageFolder(root=train_dir, transform=train_tfms)
    val_ds   = datasets.ImageFolder(root=val_dir,   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(root=test_dir,  transform=eval_tfms)

    class_to_idx = train_ds.class_to_idx

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    return loaders, class_to_idx, img_size
