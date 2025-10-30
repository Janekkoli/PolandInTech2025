import torch
from torch import nn, optim
from dataset import SentencePackDataset
from model import SentenceEmbedingClosnesnessModel, MySentenceEmbedding
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, optimizer, criterion, device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            yt = self.model(x)
            
            loss = self.criterion(yt, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def eval_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            loss = self.criterion(self.model(x), y)
            total_loss += loss.item() * x.size(0)
        return total_loss / len(loader.dataset)
    

# 1. Initialize model, loss, optimizer
model = SentenceEmbedingClosnesnessModel()
criterion = nn.BCEWithLogitsLoss()   # or MSELoss, depending on target
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10
batch_size = 32

trainer = Trainer(model, optimizer, criterion)
dataset = SentencePackDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    train_loss = trainer.train_epoch(dataloader)
    val_loss = trainer.eval_epoch(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")