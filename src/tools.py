import os
import torch
from PIL import Image
from torchvision import transforms
from crewai.tools import BaseTool
from src.model import DocumentClassifier


# --- Chargement du modèle UNE SEULE FOIS au démarrage ---
# FIX : avant, le modèle était rechargé à chaque appel → très lent
_CLASSES = ['budget', 'email', 'form', 'handwritten', 'invoice', 'letter', 'memo', 'news_article', 'resume']
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None

def _load_model():
    """Charge le modèle en mémoire une seule fois"""
    global _MODEL
    if _MODEL is not None:
        return _MODEL  # déjà chargé, on réutilise

    model = DocumentClassifier(num_classes=len(_CLASSES)).to(_DEVICE)

    if os.path.exists("models/document_classifier.pth"):
        model.load_state_dict(
            torch.load("models/document_classifier.pth", map_location=_DEVICE)
        )
        print("✓ Modèle chargé depuis models/document_classifier.pth")
    else:
        print("⚠ Modèle non trouvé — utilisation des poids aléatoires (lance train.py d'abord !)")

    model.eval()
    _MODEL = model
    return _MODEL


# Prétraitement identique à celui utilisé pendant l'entraînement
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class DocClassificationTool(BaseTool):
    name: str = "Document_Classifier_Tool"
    description: str = (
        "Analyse l'image d'un document et retourne sa catégorie "
        "(budget, email, form, handwritten, invoice, letter, memo, news_article, resume) "
        "ainsi qu'un score de confiance entre 0 et 1."
    )

    def _run(self, image_path: str) -> str:
        model = _load_model()

        try:
            # Ouvrir et prétraiter l'image
            img = Image.open(image_path).convert('RGB')
            img_tensor = _TRANSFORM(img).unsqueeze(0).to(_DEVICE)

            # Prédiction
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            category = _CLASSES[predicted_idx.item()]
            confidence_score = round(confidence.item(), 4)

            return (
                f"Catégorie : {category} | "
                f"Confiance : {confidence_score} | "
                f"Fichier : {image_path}"
            )

        except FileNotFoundError:
            return f"ERREUR : fichier introuvable → {image_path}"
        except Exception as e:
            return f"ERREUR lors de la classification : {str(e)}"
import pytesseract
from PIL import Image
from crewai.tools import BaseTool

# Chemin vers Tesseract sur Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class DocOCRTool(BaseTool):
    name: str = "Document_OCR_Tool"
    description: str = (
        "Lit le contenu textuel d'un document image (.tif, .jpg, .png) "
        "et retourne le texte extrait par OCR."
    )

    def _run(self, image_path: str) -> str:
        try:
            img = Image.open(image_path).convert('RGB')
            text = pytesseract.image_to_string(img, lang='eng')
            if text.strip():
                return f"Texte extrait :\n{text[:2000]}"  # limite à 2000 caractères
            else:
                return "OCR : aucun texte détecté dans ce document."
        except FileNotFoundError:
            return f"ERREUR : fichier introuvable → {image_path}"
        except Exception as e:
            return f"ERREUR OCR : {str(e)}"        

class FileWriterTool(BaseTool):
    name: str = "file_writer_tool"
    description: str = "Sauvegarde le rapport final sur le disque."
   
    def _run(self, filename: str, content: str) -> str:
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            return f"✅ Succès : Rapport écrit dans {filename}"
        except Exception as e:
            return f"❌ Erreur : {str(e)}"        
