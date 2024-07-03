import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from tqdm import tqdm
from transformers import DistilBertTokenizer

from src.model.loss_function import symmetric_loss
from src.model.model import CLIP
from src.validation import validation


def train(train_set, val_set, cfg):

    model = CLIP(cfg=cfg)

    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=cfg['train']['lr'], betas=(0.9, 0.98), eps=1e-6, weight_decay=cfg['train']['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     'min', 
                                                     patience=1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    if cfg['show_model_summary']:
        summary(model, (3, 224, 224))

    if cfg['train']['train_subset']:
        subset_indices = torch.randperm(len(train_set))[:cfg['train']['train_subset']]
        train_set = Subset(train_set, subset_indices)

    train_dataloader = DataLoader(
        train_set, batch_size=cfg['train']['batch_size'], shuffle=True)

    num_epochs = cfg['train']['epochs']

    if cfg['train']['continue_training']:
        model_weights = torch.load(cfg['train']['weights_path'],
                                   map_location=torch.device(device))
        model.load_state_dict(model_weights)

        # get the saved epochs and continue training from there
        last_epoch = int(cfg['train']['weights_path'].split("-")[-1].split(".")[0])
        num_epochs -= last_epoch
    last_epoch = last_epoch if cfg['train']['continue_training'] else 0

    # dataset = cfg['dataset']['dataset']
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    for epoch in range(last_epoch, last_epoch + num_epochs):
        print(f"Epoch {epoch + 1}:")
        # for i in tqdm(range(X.shape[0])):
        with tqdm(train_dataloader) as tepoch:
            for imgs, labels in tepoch:

                labels = tokenizer(labels,
                                   padding=True,
                                   truncation=True,
                                   max_length=76,
                                   return_tensors="pt")

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
        if epoch % cfg['train']['save_checkpoint_interval'] == 0:
            torch.save(model.state_dict(),
                       f"{cfg['save_model_path']}-epochs-{epoch}.pt")

    print("training done")
    torch.save(model.state_dict(),
               f"{cfg['save_model_path']}-epochs-{cfg['train']['epochs']}.pt")

    return model


def save_sets(dataset, train_set, val_set, test_set, save_path):
    """
    Save the train, val and test sets to csv files to reduce data leakage
    """

    dataset.captions.iloc[train_set.indices].to_csv(
        os.path.join(save_path, "train_set.csv"), index_label="index")
    dataset.captions.iloc[val_set.indices].to_csv(
        os.path.join(save_path, "val_set.csv"), index_label="index")
    dataset.captions.iloc[test_set.indices].to_csv(
        os.path.join(save_path, "test_set.csv"), index_label="index")


if __name__ == "__main__":

    torch.manual_seed(42)

    import math

    from src.data_processing.dataset import FlickrDataset

    cfg = {"save_model_path": "model_weights/clip-weights",
           'show_model_summary': False,
           'train': {"epochs": 100, 'lr': 5e-5,
                     'weight_decay': 0.2, "batch_size": 16,
                     "train_subset": False, 
                     "val_subset": False, # other 400
                     "save_sets": "./data",
                     'save_checkpoint_interval': 10,
                     'continue_training': False,
                     'weights_path': "model_weights/clip-weights-epochs-30.pt"},
           'dataset': {"dataset": "flickr"},
           'model': {"model_name": "clip",
                     "projections": 768}}

    # dataset = UnsplashDataset(tokenizer, "../data/unsplash/photos.tsv*")
    # dataset = FlickrDataset(image_folder_path="../data/flickr-dataset/Images/",
    #                         caption_path="../data/flickr-dataset/captions.txt")
    dataset = FlickrDataset(image_folder_path="/content/flickr-dataset/Images/",
                            caption_path="/content/flickr-dataset/captions.txt")
    
    dataset.captions['caption'] = dataset.captions.groupby(['image'])['caption'].transform(lambda x : '. '.join(x))
    dataset.captions = dataset.captions.drop_duplicates(subset = ['image'])
    dataset.captions.reset_index(drop=True, inplace= True)
    dataset_len = len(dataset)

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, lengths=[math.floor(
        dataset_len*0.8), math.ceil(dataset_len*0.1), math.floor(dataset_len*0.1)])

    # save_sets(dataset, train_set, val_set, test_set,
    #           save_path=cfg['train']['save_sets'])

    save_sets(dataset, train_set, val_set, test_set,
              save_path="/content/data")

    train(train_set=train_set, val_set=val_set, cfg=cfg)

    # cannot use FashionMNIST because size needs to be 224x224x3 at the very least
    # train_set, test_set = get_load_data(root = "../data", dataset = "FashionMNIST")
    # train(epochs = 1, train_set = train_set, in_channels = 1, num_classes = 10)
