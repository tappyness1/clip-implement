from torch.nn import Conv2d, ReLU, MaxPool2d, ConvTranspose2d, BatchNorm2d, AdaptiveAvgPool2d, Linear, Softmax
import torch.nn as nn
import torch
import numpy as np
import timm
from transformers import DistilBertModel

def get_norm_embedding(img_emb, txt_emb):
    img_emb = img_emb / img_emb.norm(dim=1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=1, keepdim=True)

    return img_emb, txt_emb    
    
class Projection(nn.Module):
    def __init__(self, input_dim, hidden_dim = 512, output_dim = 512):
        
        super(Projection, self).__init__()

        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, output_dim)
        self.relu = ReLU()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.relu(out)

        return out
    


class ImageEncoder(nn.Module):
    """
    takes in 224x224x3 image and outputs 768 dim vector. Uses the VIT based model as trained for CLIP by OpenAI

    Args:
        nn (_type_): _description_
    """
    def __init__(self, model_name='vit_base_patch16_clip_224.openai'):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained = True, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)
    
class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=True):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
    
class CLIP(nn.Module):

    def __init__(self, cfg = None):
        super().__init__()
        self.img_encoder = ImageEncoder()
        self.txt_encoder = TextEncoder()
        self.projection = Projection(cfg['model']['projections'])
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)
    
    def forward(self, img, txt):
        img_features = self.img_encoder(img)
        txt_features = self.txt_encoder(txt['input_ids'], txt['attention_mask'])   
        img_emb = self.projection(img_features)
        txt_emb = self.projection(txt_features)

        img_norm, txt_norm = get_norm_embedding(img_emb, txt_emb)
        logits = (img_norm @ txt_norm.T) / torch.exp(self.temperature)
        
        return img_emb, txt_emb, logits

if __name__ == "__main__":


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

    clip_model = CLIP()
    img_emb, txt_emb, logits = clip_model(img, label)
    print (logits)