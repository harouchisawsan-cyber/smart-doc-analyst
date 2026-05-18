import os
import json
from datetime import datetime
from PIL import Image

def logger_json(agent_name, action, result):
    """Logging structuré en JSON"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "action": action,
        "result": result
    }
    with open("logs.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def prepare_dataset(source_path, output_path, size=(224, 224)):
    """Prépare et normalise les images pour le modèle PyTorch"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for category in os.listdir(source_path):
        cat_path = os.path.join(source_path, category)
        if not os.path.isdir(cat_path): continue
        
        target_cat_path = os.path.join(output_path, category)
        os.makedirs(target_cat_path, exist_ok=True)
        
        print(f"--- Processing {category} ---")
        for img_name in os.listdir(cat_path):
            try:
                with Image.open(os.path.join(cat_path, img_name)) as img:
                    img = img.convert('RGB').resize(size)
                    img.save(os.path.join(target_cat_path, img_name.replace('.tif', '.jpg')), 'JPEG')
            except Exception as e:
                print(f"Erreur sur {img_name}: {e}")

if __name__ == "__main__":
    # À lancer pour préparer la data proprement
    prepare_dataset("data/raw/test", "data/processed")