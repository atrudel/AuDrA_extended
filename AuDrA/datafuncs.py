from __future__ import print_function

import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class ImageFolderWithRatings(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, ratings):
        super().__init__(root, transform)
        self.ratings = ratings
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithRatings, self).__getitem__(index)
        path = self.imgs[index][0]
        fname = path.split("/")[-1]
        fname = fname.split(".")[0]
        rating = self.ratings[np.where(self.ratings[:, 0] == fname)][0][1]
        tuple_with_rating= (original_tuple + (rating,))
        return tuple_with_rating


class ImageFolderWithRatingsAndFilenames(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, ratings):
        super().__init__(root, transform)
        self.ratings = ratings
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithRatingsAndFilenames, self).__getitem__(index)
        path = self.imgs[index][0]
        fname = path.split("/")[-1]
        imgname = fname.split(".")[0]
        rating = self.ratings[np.where(self.ratings[:, 0] == imgname)][0][1]
        new_tuple = (original_tuple + (rating,fname,))
        return new_tuple


class CustomDataset(Dataset):
    def __init__(self, directories: Union[str, List[str]], transform):
        directories = [directories] if not isinstance(directories, list) else directories
        self.transform = transform
        self.all_imgs = sum([
            list(Path(directory).rglob('*.jpg'))
            for directory in directories
        ], start=[])
        self.total_imgs = sorted(self.all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        img_name = str(self.total_imgs[idx])
        path_plus_image = img_name, tensor_image
        return path_plus_image


def get_subset(indices, start, end):
    return indices[start : start + end]
