# Smart Document Analyst

## 📂 Dataset
Pour ce projet, nous utilisons le dataset **RVL-CDIP**.
- **Taille :** 40,000 images (Test set utilisé pour le prototype).
- **Source :** [Kaggle - RVL-CDIP Dataset](https://www.kaggle.com/datasets/pdavpoojan/the-rvlcdip-dataset-test)
- **Localisation :** Les données doivent être extraites dans `data/raw/test/`.

## 🚀 Comment lancer le projet
1. Cloner le dépôt.
2. Installer les dépendances : `pip install -r requirements.txt`.
3. Télécharger le dataset depuis Kaggle et le placer dans le dossier `data/`.
4. Lancer l'entraînement : `python -m src.train`.