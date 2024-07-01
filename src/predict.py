import os

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import DistilBertTokenizer

from src.model.model import get_norm_embedding


def encode_image(model, imgs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    with torch.no_grad():  # otherwise boom cuda issue

        image_features = model.img_encoder(imgs.to(device))
        image_embeddings = model.projection(image_features)

    return image_embeddings


def encode_text_query(model, text, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    with torch.no_grad():  # otherwise boom cuda issue
        text = tokenizer(text, padding=True, truncation=True,
                         max_length=76, return_tensors="pt")
        text_features = model.txt_encoder(text['input_ids'].to(
            device), text['attention_mask'].to(device))
        text_embedding = model.projection(text_features)

    return text_embedding


def find_similar(image_embeddings, text_embedding, top_k):
    final_image_embeddings = image_embeddings[:, 1:]
    final_image_embeddings, text_embedding = get_norm_embedding(
        final_image_embeddings, text_embedding)
    similarities = (final_image_embeddings @ text_embedding.T).squeeze(1)
    similar_idx = similarities.argsort(descending=True)
    return similar_idx[:top_k]


class ImageSearch:

    def __init__(self, model, dataset):
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset

    def get_embeddings(self, dataloader, subset_indices):
        to_cat = torch.zeros(1, 512)

        with tqdm(dataloader) as tepoch:
            for imgs, _ in tepoch:
                encoded_images = encode_image(self.model, imgs)
                to_cat = torch.cat((to_cat, encoded_images.to('cpu')), 0)
        final_embedding = to_cat[1:]
        # concat the subset_index to the front of the embeddings
        self.final_embedding = torch.cat((torch.Tensor(subset_indices).reshape(
            (len(self.dataset), 1)), final_embedding), dim=1)

    def show_images(self, prompt, dataset_df, img_path="/content/flickr-dataset/Images", top_k=6):

        tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased")
        text_embedding = encode_text_query(self.model, prompt, tokenizer)
        top_k_similar = find_similar(self.final_embedding.to(
            self.device), text_embedding, top_k)
        top_k_image_ids = self.final_embedding[:, 0][top_k_similar.cpu(
        ).numpy().tolist()].numpy().astype(int).tolist()
        results = dataset_df.loc[top_k_image_ids]

        image_files = results['image'].tolist()

        imgs = np.zeros((1, 224, 224, 3))
        for i in range(len(image_files)):
            img = Image.open(os.path.join(img_path, image_files[i]))
            img = np.array(img.resize((224, 224))).reshape(1, 224, 224, 3)
            # img = img.unsqueeze(0)
            imgs = np.vstack((imgs, img))
        imgs = imgs[1:]
        fig = px.imshow(np.array(imgs), facet_col=0, facet_col_wrap=top_k /
                        2, width=900, height=600, title=f"Prompt: {prompt}")
        fig.show()


if __name__ == "__main__":
    img_search = ImageSearch(model=None, dataset=None)
