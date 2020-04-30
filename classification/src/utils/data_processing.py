import cv2

import torch
import torchvision
from albumentations import (
    Blur,
    ChannelDropout,
    Compose,
    Flip,
    GaussNoise,
    HueSaturationValue,
    Normalize,
    RandomBrightnessContrast,
    RandomGamma,
    RandomGridShuffle,
    RandomResizedCrop,
    RandomRotate90,
    Resize,
    RGBShift,
    Transpose,
)
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.datasets.folder import IMG_EXTENSIONS


def _albumentations_loader(path):
    """Load an image."""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class AlbumentationsImageFolder(torchvision.datasets.DatasetFolder):
    """Helper class to apply augmentations and form the dataset."""

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=_albumentations_loader,
        is_valid_file=None,
    ):
        super(AlbumentationsImageFolder, self).__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            augmented = self.transform(image=sample)
            sample = augmented["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


def train_val_dataloaders(
    train_path: str, val_path: str, augment: bool, batch_size: int
):
    """Form the dataloaders for training and validation and store them in the dictionary.

    :param train_path: path to images for trainin
    :param val_path: path to images for validation
    :param batch_size: size of the batch
    :return: the dictionary with dataloaders
    """
    if augment:
        train_transform = Compose(
            [
                Blur(p=0.1),
                ChannelDropout(p=0.1),
                Flip(p=0.5),
                GaussNoise((10.0, 30.0), 25.0, p=0.1),
                HueSaturationValue(p=0.1),
                RandomBrightnessContrast(brightness_limit=(-0.20, 0.50), p=0.1),
                RandomGamma(p=0.1),
                RandomGridShuffle(p=0.1),
                RandomRotate90(p=0.5),
                RGBShift(p=0.1),
                Transpose(p=0.25),
                RandomResizedCrop(height=224, width=224, p=1.0),
                Normalize(),
                ToTensorV2(),
            ]
        )
    else:
        train_transform = Compose([Resize(224, 224), Normalize(), ToTensorV2()])

    val_transforms = Compose([Resize(224, 224), Normalize(), ToTensorV2()])

    train_dataset = AlbumentationsImageFolder(train_path, train_transform)
    val_dataset = AlbumentationsImageFolder(val_path, val_transforms)

    dataloader = dict()

    dataloader["train"] = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    dataloader["val"] = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    return dataloader
