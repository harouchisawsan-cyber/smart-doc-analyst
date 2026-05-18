import os

def check_data_folders():
    base_path = "data/raw/test"
    if os.path.exists(base_path):
        classes = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        print(f"Dataset prêt avec {len(classes)} classes.")