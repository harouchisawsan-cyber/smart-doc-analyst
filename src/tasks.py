#Contenu : La description précise de ce que chaque agent doit faire.
#Rôle : "La tâche 1 est de classer l'image", "La tâche 2 est d'extraire le texte", etc.
from crewai import Task

class DocumentTasks:
    def classification_task(self, agent, image_path):
        return Task(
            description=f"Analyser l'image située à {image_path} et déterminer sa catégorie.",
            expected_output="Le nom de la catégorie du document (ex: 'invoice').",
            agent=agent
        )

    def extraction_task(self, agent, output_file):
        return Task(
            description="Lire le contenu du document classifié et extraire les informations clés en format JSON.",
            expected_output="Un objet JSON contenant les champs pertinents (date, montant, émetteur, etc.).",
            agent=agent,
            output_file=output_file
        )