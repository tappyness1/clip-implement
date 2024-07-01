import numpy as np
import torch
from torch.nn import CrossEntropyLoss, LogSoftmax
from torch.nn import functional as F

from src.model.model import CLIP


def symmetric_loss(logits):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    labels = torch.arange(logits.shape[0])
    labels = labels.to(device)
    loss = CrossEntropyLoss()
    loss_i = loss(logits, labels)
    loss_t = loss(logits.T, labels)
    loss = (loss_i + loss_t) / 2.0
    return loss

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import DistilBertTokenizer

    from src.data_processing.dataset import UnsplashDataset
    from src.model import CLIP

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
