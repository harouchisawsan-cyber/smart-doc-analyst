import os
import json
from datetime import datetime
from PIL import Image


# --- LOGGING JSON ---

LOG_FILE = "logs/agent_logs.json"

def setup_logger():
    """Crée le dossier logs/ si il n'existe pas"""
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump([], f)  # fichier JSON vide au départ


def log_action(agent_name: str, action: str, result: str, status: str = "success"):
    """
    Enregistre une action agent dans logs/agent_logs.json
    
    Exemple d'entrée :
    {
        "timestamp": "2025-05-20T14:32:01.123456",
        "agent": "Classificateur Visuel",
        "action": "classify_document",
        "result": "Catégorie : invoice | Confiance : 0.94",
        "status": "success"
    }
    """
    setup_logger()

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "action": action,
        "result": str(result),
        "status": status
    }

    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        logs = []

    logs.append(log_entry)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


def log_human_checkpoint(image_path: str, reason: str):
    """Log spécial quand le superviseur demande une validation humaine"""
    log_action(
        agent_name="Superviseur Qualité",
        action="human_checkpoint_triggered",
        result=f"Fichier: {image_path} | Raison: {reason}",
        status="waiting_human"
    )
    print(f"\n{'='*60}")
    print("  ⚠  VALIDATION HUMAINE REQUISE")
    print(f"  Fichier  : {image_path}")
    print(f"  Raison   : {reason}")
    print(f"{'='*60}\n")


def get_logs(last_n: int = None):
    """
    Retourne les logs — utile pour débugger ou afficher l'historique
    
    Exemple : get_logs(last_n=5) → les 5 dernières entrées
    """
    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
        return logs[-last_n:] if last_n else logs
    except (FileNotFoundError, json.JSONDecodeError):
        return []



def prepare_dataset(source_path: str, output_path: str, size: tuple = (224, 224)):
    """
    Convertit les images .tif du dataset RVL-CDIP en .jpg redimensionnés
    Lance cette fonction UNE SEULE FOIS avant d'entraîner le modèle
    """
    os.makedirs(output_path, exist_ok=True)
    total = 0
    errors = 0

    for category in os.listdir(source_path):
        cat_path = os.path.join(source_path, category)
        if not os.path.isdir(cat_path):
            continue

        target_cat_path = os.path.join(output_path, category)
        os.makedirs(target_cat_path, exist_ok=True)

        print(f"--- Traitement : {category} ---")

        for img_name in os.listdir(cat_path):
            output_name = os.path.splitext(img_name)[0] + '.jpg'
            output_file = os.path.join(target_cat_path, output_name)

            # FIX : on saute les images déjà traitées
            if os.path.exists(output_file):
                continue

            try:
                with Image.open(os.path.join(cat_path, img_name)) as img:
                    img = img.convert('RGB').resize(size)
                    img.save(output_file, 'JPEG', quality=95)
                    total += 1
            except Exception as e:
                print(f"  Erreur sur {img_name} : {e}")
                errors += 1

    print(f"\n✓ Dataset prêt : {total} images converties | {errors} erreurs")
    log_action("System", "prepare_dataset", f"{total} images converties, {errors} erreurs")


if __name__ == "__main__":
    prepare_dataset("data/raw", "data/processed")
