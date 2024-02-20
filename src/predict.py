import torch.nn as nn
import torch
import pandas as pd

def encode_image(model, imgs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    image_features = model.img_encoder(imgs.to(device))
    image_embeddings = model.projection(image_features)

    return image_embeddings
