import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd
from src.loss_function import symmetric_loss
from sklearn.metrics import confusion_matrix

def validation(model, val_set, tokenizer, cfg):
    """Simple validation workflow. Current implementation is for F1 score

    Args:
        model (_type_): _description_
        val_set (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.eval()

    if cfg['train']['subset']:
        subset_indices = torch.randperm(len(val_set))[:cfg['train']['subset']]
        val_set = Subset(val_set, subset_indices)
    val_dataloader = DataLoader(val_set, batch_size=cfg['train']['batch_size'], shuffle = True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    losses = []

    with tqdm(val_dataloader) as tepoch:

        for imgs, labels in tepoch:
            
            labels = tokenizer(labels, padding=True, truncation=True, max_length = 76, return_tensors="pt")
            labels = labels.to(device)
            imgs = imgs.to(device)
            
            _, _, logits = model(imgs, labels)
            loss = symmetric_loss(logits)
            tepoch.set_postfix(loss=loss.item())  
            losses.append(loss.item())

    print (f"Validation Loss: {sum(losses)/len(losses)}")

    return sum(losses)/len(losses)


if __name__ == "__main__":
    
    from src.dataset import get_load_data


    cfg = {"save_model_path": "model_weights/model_weights.pt",
           'show_model_summary': False, 
           'train': {"epochs": 3, 'lr': 5e-5, 'weight_decay': 0.2, "batch_size": 16},
           'dataset': {"dataset": "unsplash"},
           'model':{"projections": 768}}
    
    _, val_set = get_load_data(root = "../data", dataset = cfg['dataset']['dataset'])

    trained_model_path = "model_weights/model_weights.pt"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(trained_model_path, map_location=torch.device(device))
    validation(model, val_set, cfg_obj=cfg)
            