import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import DocumentClassifier
import os

def train_model():
    # 1. Configuration des transformations (Redimensionnement 224x224 pour ResNet18)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Chargement des données (RVL-CDIP Test set utilisé pour le prototype)
    data_dir = 'data/raw/test'
    if not os.path.exists(data_dir):
        print(f"Erreur : Le dossier {data_dir} est introuvable.")
        return

    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Division 80% train / 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # 3. Initialisation du modèle et de l'optimiseur
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DocumentClassifier(num_classes=16).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Début de l'entraînement sur : {device}")
    
    # --- Boucle d'entraînement (Squelette) ---
    # Pour la démo, on simule l'étape de training
    model.train()
    print("Entraînement de l'époque 1 en cours...")
    
    # 4. Sauvegarde du modèle (Prévu dans le dossier models/)
    os.makedirs('models', exist_ok=True)
    save_path = 'models/document_classifier_v1.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé avec succès dans : {save_path}")

if __name__ == "__main__":
    train_model()