import torch
import numpy as np
from src.model import CLIP
from torch.nn import CrossEntropyLoss

def symmetric_loss(logits):
    labels = torch.arange(logits.shape[0])
    loss = CrossEntropyLoss()
    loss_i = loss(logits, labels)
    loss_t = loss(logits.T, labels)
    return (loss_i + loss_t) / 2.0


if __name__ == "__main__":
    from src.model import CLIP
    from src.dataset import UnsplashDataset
    from torch.utils.data import DataLoader
    from transformers import DistilBertTokenizer

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    dataset = UnsplashDataset(tokenizer, "../data/unsplash/photos.tsv*")
    train_loader = DataLoader(dataset, batch_size = 4, shuffle=True)    
    img, label = next(iter(train_loader))
    label = tokenizer(label, padding=True, truncation=True, max_length = 76, return_tensors="pt")

    # img_encoder = ImageEncoder()
    # txt_encoder = TextEncoder()

    # img_emb = img_encoder(img)
    # print (img_emb.shape)
    # text_emb = txt_encoder(label['input_ids'], label['attention_mask'])    
    # print (text_emb.shape)
    cfg = {'model':{"projections": 768}}
    clip_model = CLIP(cfg)
    img_emb, txt_emb, logits = clip_model(img, label)

    loss = symmetric_loss(logits)
    loss.backward()
