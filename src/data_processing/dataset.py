import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import Compose, PILToTensor, Resize, ToTensor


def get_photos_tsv(path):
    files = glob.glob(path)
    subsets = []
    for filename in files:
        subsets.append(pd.read_csv(filename, sep='\t', header=0))
    return pd.concat(subsets, axis=0, ignore_index=True)


class UnsplashDataset(Dataset):
    def __init__(self, tokenizer, photos_tsv_path="../data/unsplash/photos.tsv*",):
        self.photos_df = get_photos_tsv(photos_tsv_path)
        self.img_urls = self.photos_df['photo_image_url']
        self.img_desc = self.photos_df['photo_description']
        self.img_desc_ai = self.photos_df['ai_description']

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((512, 512))
                                             ])
        # self.target_transform = target_transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.img_urls)

    def __getitem__(self, idx):
        url = self.img_urls[idx]
        img_desc = self.img_desc[idx]
        img_desc_ai = self.img_desc_ai[idx]

        image = Image.open(requests.get(url, stream=True).raw)
        if str(img_desc) == "nan":
            label = img_desc_ai
        else:
            label = img_desc

        image = self.transform(image)

        # somehow the transforms does not resize
        image = transforms.functional.resize(image, (224, 224))
        label = self.tokenizer(
            label, padding=True, truncation=True, max_length=76, return_tensors="pt")

        return image, label


class FlickrDataset(Dataset):
    def __init__(self, image_folder_path, caption_path):
        self.image_folder_path = image_folder_path

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((512, 512))
                                             ])
        self.captions = pd.read_csv(caption_path, sep=',', header=0)

    def __len__(self):
        return self.captions.shape[0]

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder_path,
                                self.captions.iloc[idx, 0])

        image = Image.open(img_name)
        label = self.captions.iloc[idx, 1]

        image = self.transform(image)

        # somehow the transforms does not resize
        image = transforms.functional.resize(image, (224, 224))
        return image, label


if __name__ == "__main__":
    pass
    # train, test = get_load_data(root = "../data")
    # img, label = train[1]
    # plt.imshow(img.squeeze(), cmap="gray")
    # plt.show()

    # # for local testing
    # train, test = get_load_data(root = "../data", dataset = "VOCSegmentation", download = False)
    # img, smnt = train[12]
    # print (smnt)

    # # for gcp or whatever
    # train, test = get_load_data(root = "../data", dataset = "FashionMNIST", download = True)
