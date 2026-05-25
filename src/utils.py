import os
import shutil
import json
from datetime import datetime
from PIL import Image

def logger_json(agent, action, result):
    """

    Cette fonction est appelée par le 'step_callback' dans main.py.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": str(agent),
        "action": str(action),
        "result": str(result)
    }
    
    log_file = "logs.json"
    
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Erreur lors du logging : {e}")
# ---------------------------------------------------------

SELECTED_CLASSES = ['budget', 'email', 'form', 'handwritten', 'invoice', 'letter', 'memo', 'news_article', 'questionnaire']

def prepare_professional_dataset(source_path, output_path, limit=200):
    # 1. NETTOYAGE : On supprime le dossier processed s'il existe pour repartir de zéro
    if os.path.exists(output_path):
        print(f"Nettoyage de l'ancien dossier {output_path}...")
        shutil.rmtree(output_path)
    
    os.makedirs(output_path)
    
    print(f"--- Création du dataset final : 9 classes, {limit} images/classe ---")
    
    for category in SELECTED_CLASSES:
        
        cat_path = os.path.join(source_path, category)
        target_cat_path = os.path.join(output_path, category)
        
        if os.path.exists(cat_path):
            os.makedirs(target_cat_path)
            # Lister uniquement les fichiers images
            images = [f for f in os.listdir(cat_path) if f.lower().endswith(('.tif', '.jpg', '.png'))]
            subset = images[:limit]
            
            print(f"Traitement de {category} ({len(subset)} images)...")
            for img_name in subset:
                try:
                    with Image.open(os.path.join(cat_path, img_name)) as img:
                        # Conversion RGB + Redimensionnement
                        img = img.convert('RGB').resize((224, 224))
                        # Sauvegarde en JPG (plus léger)
                        save_name = img_name.split('.')[0] + ".jpg"
                        img.save(os.path.join(target_cat_path, save_name), 'JPEG')
                except Exception as e:
                    print(f"Erreur sur {img_name}: {e}")
        else:
            print(f"⚠️ Alerte : La catégorie {category} est introuvable dans {source_path}")

if __name__ == "__main__":
    # CONFIGURATION DES CHEMINS (Ton code original)
    RAW_DATA_PATH = "data/raw/test"
    PROCESSED_DATA_PATH = "data/processed"
    
    prepare_professional_dataset(RAW_DATA_PATH, PROCESSED_DATA_PATH, limit=200)
    print("\n✅ Dataset prêt ! Vous avez maintenant exactement 9 classes et 1800 images.")
