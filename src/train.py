import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import DocumentClassifier


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

    return running_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    """Évalue le modèle sur le val set — NOUVEAU"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")

    # --- FIX : on crée le dossier models/ s'il n'existe pas ---
    os.makedirs("models", exist_ok=True)

    # Data Augmentation (train uniquement)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Pas d'augmentation sur la validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Charger le dataset complet avec train_transform d'abord
    full_dataset = datasets.ImageFolder('data/processed', transform=train_transform)

    # Split train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # reproductible
    )

    # Appliquer val_transform sur le val set
    from torch.utils.data import Subset
    val_ds = Subset(
      datasets.ImageFolder('data/processed', transform=val_transform),
      val_ds.indices
    )

    print(f"Train : {train_size} images | Val : {val_size} images")
    print(f"Classes : {full_dataset.classes}")

    num_classes = len(full_dataset.classes)

   # D'abord limiter les données
    from torch.utils.data import Subset
    train_ds = Subset(train_ds, range(100))   # au lieu de range(10000)
    val_ds = Subset(val_ds, range(20))         # 20% de 100
    NUM_EPOCHS = 10                            # au lieu de 3

# Ensuite créer les loaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = DocumentClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Scheduler : réduit le LR si la val_loss stagne
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        # --- FIX : on sauvegarde uniquement le meilleur modèle ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/document_classifier.pth")
            print(f"  ✓ Meilleur modèle sauvegardé (val_acc={val_acc:.2f}%)")

    print(f"\nEntraînement terminé. Meilleure val accuracy : {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
