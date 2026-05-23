import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from src.model import DocumentClassifier

# Config
CLASSES = ['budget', 'email', 'form', 'handwritten', 'invoice', 'letter', 'memo', 'news_article', 'resume']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle
model = DocumentClassifier(num_classes=len(CLASSES)).to(DEVICE)
model.load_state_dict(torch.load("models/document_classifier.pth", map_location=DEVICE))
model.eval()

# Charger le val set
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset = datasets.ImageFolder('data/processed', transform=val_transform)
indices = []
class_counts = {}
for idx, (_, label) in enumerate(dataset.imgs):
    if class_counts.get(label, 0) < 20:
        indices.append(idx)
        class_counts[label] = class_counts.get(label, 0) + 1
    if len(indices) >= 180:
        break
val_ds = Subset(dataset, indices)  
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

# Prédictions
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix — Document Classifier')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("✓ confusion_matrix.png sauvegardée !")

# Rapport de classification
unique_labels = sorted(set(all_labels))
present_classes = [CLASSES[i] for i in unique_labels]
print("\n" + classification_report(all_labels, all_preds))
