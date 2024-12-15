import torch
from model import DNN
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim import Adagrad
from torch.optim import SGD
from adabelief_pytorch import AdaBelief

def train(train_data, val_data, batch_size, epochs, optimizer_name:str):
    model = DNN(train_data.data.shape[1])
    criterion = CrossEntropyLoss()
    if optimizer_name.lower() == "adamw":
        optimizer = AdamW(model.parameters(), lr=0.002)
    elif optimizer_name.lower() == "adabelief":
        optimizer = AdaBelief(model.parameters(), lr=0.01)
    elif optimizer_name.lower() == "sgd":
        optimizer = SGD(model.parameters(), lr=0.3)
    elif optimizer_name.lower() == "adagrad":
        optimizer = Adagrad(model.parameters(), lr=0.02)
        
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=val_data.data.shape[0], shuffle=True)
    
    print(f"\nStarting training with {optimizer_name} optimizer\n")
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch_idx, (batch, labels) in enumerate(train_dataloader):
            y_pred = model(batch)
            train_loss = criterion(y_pred, labels)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            for batch, labels in val_dataloader:
                val_y_pred = model(batch)
                val_loss = criterion(val_y_pred, labels)
                print(f"EPOCH {epoch}, VAL_LOSS {val_loss.item()} -> TRAIN_LOSS {sum(train_losses)/len(train_losses)}")
    return model