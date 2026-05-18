#Contenu : La description précise de ce que chaque agent doit faire.
#Rôle : "La tâche 1 est de classer l'image", "La tâche 2 est d'extraire le texte", etc.
from crewai import Task

class DocumentTasks:
    def classification_task(self, agent, image_path):
        return Task(
            description=f"Analyse l'image du document située à : {image_path}. Utilise ton outil de classification pour dire de quelle catégorie il s'agit.",
            expected_output="Le nom de la catégorie (ex: invoice, email, letter).",
            agent=agent
        )

    def extraction_task(self, agent):
        return Task(
            description="À partir du document classifié, analyse le contenu textuel et extrait les informations clés (Date, Montant, Nom de l'entreprise) sous forme de JSON structuré.",
            expected_output="Un objet JSON propre contenant les informations extraites.",
            agent=agent
        )