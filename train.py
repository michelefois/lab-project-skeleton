# Training loop
import os
import wandb
import torch
import torch.nn as nn

from models.customnet_lab2 import CustomNet
from data.dataloader_lab2 import get_dataloaders
from eval import validate



def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # todo...
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    return train_loss, train_accuracy

def main():
    # Inizializza W&B come mostrato nelle slide
    wandb.init(project="tiny-imagenet-lab", name="customnet-run")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 10
    lr = 0.001

    train_loader, val_loader = get_dataloaders()

    # Inizializzazione modello, funzione di costo e ottimizzatore
    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    best_acc = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(epoch, model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Log metriche su W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # Salva il checkpoint migliore
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')

    print(f'Best validation accuracy: {best_acc:.2f}%')
    wandb.finish()

if __name__ == '__main__':
    main()