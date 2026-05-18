import torch
from torchvision import datasets, transforms, models

# 1. Prétraitement (Redimensionner pour que le modèle ne rame pas)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Chargement automatique (C'est ici que la magie opère)
train_data = datasets.ImageFolder('data/raw', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

# 3. Charger un modèle déjà "intelligent" (ResNet18)
model = models.resnet18(pretrained=True)

# 4. Ajuster la sortie pour tes 15 classes
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 15) 

print(f"Prêt à entraîner sur {len(train_data)} images avec 15 classes.")