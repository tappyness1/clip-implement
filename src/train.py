from src.model import CLIP
import torch.optim as optim
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from src.loss_function import symmetric_loss
from transformers import DistilBertTokenizer
from src.validation import validation 

def train(train_set, val_set, cfg):
           
    model = CLIP(cfg = cfg)

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=cfg['train']['lr'], betas=(0.9,0.98),eps=1e-6, weight_decay=cfg['train']['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    if cfg['show_model_summary']:
        summary(model, (3,224,224))

    if cfg['train']['subset']:
        subset_indices = torch.randperm(len(train_set))[:cfg['train']['subset']]
        train_set = Subset(train_set, subset_indices)

    train_dataloader = DataLoader(train_set, batch_size=cfg['train']['batch_size'], shuffle = True)
    
    # dataset = cfg['dataset']['dataset']
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    for epoch in range(cfg['train']['epochs']):
        print (f"Epoch {epoch + 1}:")
        # for i in tqdm(range(X.shape[0])):
        with tqdm(train_dataloader) as tepoch:
            for imgs, labels in tepoch:
                
                labels = tokenizer(labels, padding=True, truncation=True, max_length = 76, return_tensors="pt")
                labels = labels.to(device)
                imgs = imgs.to(device)
                
                _, _, logits = model(imgs, labels)
                loss = symmetric_loss(logits)

                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())

        loss = validation(model, val_set, tokenizer, cfg)
        scheduler.step(loss)
        model.train()
        
    print("training done")
    torch.save(model, cfg['save_model_path'])

    return model

if __name__ == "__main__":

    torch.manual_seed(42)

    from src.dataset import UnsplashDataset, FlickrDataset
    
    import math

    cfg = {"save_model_path": "model_weights/model_weights.pt",
           'show_model_summary': False, 
           'train': {"epochs": 3, 'lr': 5e-5, 'weight_decay': 0.2, "batch_size": 16},
           'dataset': {"dataset": "unsplash"},
           'model':{"projections": 768}}
    # dataset = UnsplashDataset(tokenizer, "../data/unsplash/photos.tsv*")
    dataset = FlickrDataset(image_folder_path = "../data/flickr-dataset/Images/", caption_path = "../data/flickr-dataset/captions.txt")
    dataset_len = len(dataset)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, lengths = [math.floor(dataset_len*0.8), math.ceil(dataset_len*0.1), math.floor(dataset_len*0.1)])

    train(train_set = train_set, val_set = val_set, cfg = cfg)

    # cannot use FashionMNIST because size needs to be 224x224x3 at the very least
    # train_set, test_set = get_load_data(root = "../data", dataset = "FashionMNIST")
    # train(epochs = 1, train_set = train_set, in_channels = 1, num_classes = 10)
    