#Contenu : Une classe qui charge ton modèle entraîné et qui a une fonction run(image_path).
#Rôle : Elle permet à l'agent CrewAI d'utiliser ton modèle PyTorch comme si c'était un simple outil (comme une calculatrice ou une recherche Google).
from crewai.tools import BaseTool
import torch
from PIL import Image
from torchvision import transforms
from src.model import DocumentClassifier
import os

class DocClassificationTool(BaseTool):
    name: str = "Document_Classifier_Tool"
    description: str = "Analyse l'image d'un document et retourne sa catégorie (Facture, Email, Contrat, etc.)."

    def _run(self, image_path: str) -> str:
        # 1. Liste des classes (dans l'ordre du dataset)
        classes = ['advertisement', 'budget', 'email', 'file_folder', 'form', 'handwritten', 'invoice', 'letter', 'memo', 'news_article', 'presentation', 'questionnaire', 'resume', 'scientific_publication', 'scientific_report', 'specification']

        # 2. Configuration du modèle
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DocumentClassifier(num_classes=16)
        
        # On essaie de charger les poids si le fichier existe, sinon on reste en mode 'aléatoire' pour le test
        if os.path.exists("models/document_classifier.pth"):
            model.load_state_dict(torch.load("models/document_classifier.pth", map_location=device))
        
        model.to(device).eval()

        # 3. Prétraitement de l'image entrante
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        try:
            img = Image.open(image_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            
            # 4. Prédiction
            with torch.no_grad():
                output = model(img)
                _, predicted = torch.max(output, 1)
                category = classes[predicted.item()]
            
            return f"Le document a été classifié comme : {category}"
        except Exception as e:
            return f"Erreur lors de la classification : {str(e)}"