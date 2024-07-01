import glob
import os

import numpy as np
import pandas as pd
import requests
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image


def get_photos_tsv(path):
    files = glob.glob(path)
    subsets = []
    for filename in files:
        subsets.append(pd.read_csv(filename, sep='\t', header=0))
    return pd.concat(subsets, axis=0, ignore_index=True)

def download_data(save_folder, url, photo_id):
    
    # This statement requests the resource at 
    # the given link, extracts its contents 
    # and saves it in a variable 
    data = requests.get(url).content 
    
    # Opening a new file named img with extension .jpg 
    # This file would store the data of the image file 
    f = open(f'{save_folder}{photo_id}.jpg','wb') 
    
    # Storing the image data inside the data variable to the file 
    f.write(data) 
    f.close() 


def download_all_url(photos_tsv_path = "../data/unsplash/photos.tsv*", save_folder = "../data/unsplash/images/"):
    photos_df = get_photos_tsv(photos_tsv_path)
    photos_url = photos_df['photo_image_url'].tolist()
    photos_id = photos_df['photo_id'].tolist()
    folder_files = os.listdir(save_folder)
    for ind, url in enumerate(photos_url):
        if f'{photos_id[ind]}.jpg' in folder_files:
            continue
        download_data(save_folder, url, photos_id[ind])

    print ("Done")


class UnsplashDataset(Dataset):
    def __init__(self, photos_tsv_path = "../data/unsplash/photos.tsv*", photo_folder_path = "../data/unsplash/images/"):
        self.photos_df = get_photos_tsv(photos_tsv_path)
        self.img_urls = self.photos_df['photo_image_url']
        self.img_id = self.photos_df['photo_id']
        self.img_id = photo_folder_path + self.img_id + ".jpg"
        self.img_desc = self.photos_df['photo_description']
        self.img_desc_ai = self.photos_df['ai_description']

        self.transform = transforms.Compose([transforms.ToTensor(), 
                                             transforms.Resize((512,512))
                                             ])
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.img_urls)

    def __getitem__(self, idx):
        
        img_desc = self.img_desc[idx]
        img_desc_ai = self.img_desc_ai[idx]

        # url = self.img_urls[idx]
        # image = Image.open(requests.get(url, stream=True).raw)
        photo_id = self.img_id[idx]
        image = Image.open(photo_id)
        if str(img_desc) == "nan":
            label = img_desc_ai
        else:
            label = img_desc

        image = self.transform(image)

        # somehow the transforms does not resize
        image = transforms.functional.resize(image, (512, 512))

        return image, label
    
if __name__ == "__main__": 
    
    download_all_url()