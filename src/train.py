import torch.nn as nn
from src.model import CLIP
import torch.optim as optim
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchsummary import summary
import hydra
from omegaconf import DictConfig, OmegaConf
from src.loss_function import elbo_loss
from src.dataset import get_load_data
import numpy as np

def train(train_set, cfg):
           
    model = CLIP(cfg_obj = cfg)

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    if cfg['show_model_summary']:
        summary(model, (3,224,224))

    train_dataloader = DataLoader(train_set, batch_size=5, shuffle = True)
    dataset = cfg['dataset']['dataset']
    
    for epoch in range(cfg['train']['epochs']):
        print (f"Epoch {epoch + 1}:")
        # for i in tqdm(range(X.shape[0])):
        with tqdm(train_dataloader) as tepoch:
            for imgs, classes in tepoch:
                imgs = imgs.to(device)
                
                mean, log_var, z, decoded = model(imgs)
                loss = elbo_loss(mean, log_var, z, model.log_scale, decoded, dataset, imgs)
                
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
        
    print("training done")
    torch.save(model, cfg['save_model_path'])

    return model

if __name__ == "__main__":

    torch.manual_seed(42)

    from src.dataset import UnsplashDataset
    from torch.utils.data import DataLoader
    from transformers import DistilBertTokenizer

    cfg = {"save_model_path": "model_weights/model_weights.pt",
           'show_model_summary': False, 
           'train': {"epochs": 3, 'lr': 0.005, 'weight_decay': 5e-3},
           'dataset': {"dataset": "unsplash"}}

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    dataset = UnsplashDataset(tokenizer, "../data/unsplash/photos.tsv*")

    train(train_set = train_set, cfg = cfg)

    # cannot use FashionMNIST because size needs to be 224x224x3 at the very least
    # train_set, test_set = get_load_data(root = "../data", dataset = "FashionMNIST")
    # train(epochs = 1, train_set = train_set, in_channels = 1, num_classes = 10)
    