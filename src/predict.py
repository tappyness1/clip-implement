import torch.nn as nn
import torch
import pandas as pd
from src.model import get_norm_embedding

def encode_image(model, imgs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    
    with torch.no_grad(): # otherwise boom cuda issue

        image_features = model.img_encoder(imgs.to(device))
        image_embeddings = model.projection(image_features)

    return image_embeddings

def encode_text_query(model, text, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    
    with torch.no_grad(): # otherwise boom cuda issue
        text = tokenizer(text, padding=True, truncation=True, max_length = 76, return_tensors="pt")
        text_features = model.txt_encoder(text['input_ids'].to(device), text['attention_mask'].to(device))
        text_embedding = model.projection(text_features)

    return text_embedding

def find_similar(image_embeddings, text_embedding, top_k):
    final_image_embeddings = image_embeddings[:, 1:]
    final_image_embeddings, text_embedding = get_norm_embedding(final_image_embeddings, text_embedding)
    similarities = (final_image_embeddings @ text_embedding.T).squeeze(1)
    similar_idx = similarities.argsort(descending=True)
    return similar_idx[:top_k]