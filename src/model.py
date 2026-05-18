#Contenu : La classe PyTorch (le CNN) et l'architecture de  modèle de classification.
#Rôle : C'est le plan de construction de ton IA de classification.

import torch
import torch.nn as nn
from torchvision import models

class DocumentClassifier(nn.Module):
    def __init__(self, num_classes=16):
        super(DocumentClassifier, self).__init__()
        # On charge un ResNet18 pré-entraîné
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # On remplace la dernière couche (fully connected) pour l'adapter à nos 16 classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)