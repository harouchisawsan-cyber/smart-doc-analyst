#Contenu : La classe PyTorch (le CNN) et l'architecture de  modèle de classification.
#Rôle : C'est le plan de construction de ton IA de classification.

import torch
import torch.nn as nn
from torchvision import models

class DocumentClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super(DocumentClassifier, self).__init__()
        # On charge les poids pré-entraînés pour bénéficier de l'intelligence visuelle
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # On remplace la tête du réseau pour tes 16 classes
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.2), # Pour éviter l'overfitting
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)
