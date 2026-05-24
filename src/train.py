import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import DocumentClassifier
import os

def evaluate(model, loader, criterion, device):
    """Fonction pour évaluer le modèle sur le set de validation"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return val_loss/len(loader), 100.*correct/total

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss/len(loader), 100.*correct/total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur : {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Chargement du dataset (900 images générées par utils.py)
    if not os.path.exists('data/processed'):
        print("Erreur : Lancez d'abord src/utils.py pour préparer les 9 classes.")
        return

    dataset = datasets.ImageFolder('data/processed', transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True) # Batch plus petit pour 900 images
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)


    model = DocumentClassifier(num_classes=9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Learning rate plus faible pour le fine-tuning

    # Entraînement
    epochs = 10 
    print(f"Lancement du Fine-tuning sur 9 classes...")
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.3f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.3f} Acc: {val_acc:.2f}%")
        
    # Sauvegarde sécurisée
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), "models/document_classifier.pth")
    print("\n✅ Entraînement terminé et modèle sauvegardé !")

if __name__ == "__main__":
    main()
