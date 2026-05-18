import os
from dotenv import load_dotenv
from crewai import Crew
from src.agents import DocumentAgents
from src.tasks import DocumentTasks
from src.tools import DocClassificationTool

# 1. Charger les variables d'environnement
load_dotenv()

def run_analysis(image_path):
    # Initialiser les composants
    agents_factory = DocumentAgents()
    tasks_factory = DocumentTasks()
    classifier_tool = DocClassificationTool()

    # Créer les agents
    classifier_agent = agents_factory.visual_classifier_agent()
    # On assigne l'outil au classificateur
    classifier_agent.tools = [classifier_tool]
    
    extractor_agent = agents_factory.extractor_agent()

    # Définir les tâches
    task1 = tasks_factory.classification_task(classifier_agent, image_path)
    task2 = tasks_factory.extraction_task(extractor_agent)

    # Créer et lancer la Crew
    crew = Crew(
        agents=[classifier_agent, extractor_agent],
        tasks=[task1, task2],
        verbose=True
    )

    return crew.kickoff()

if __name__ == "__main__":
    # Test sur une image réelle de ton dossier
    image_a_tester = "data/raw/test/invoice/0000023361.tif" 
    
    if os.path.exists(image_a_tester):
        print(f"--- Lancement de l'analyse pour : {image_a_tester} ---")
        result = run_analysis(image_a_tester)
        print("\n\n########################")
        print("## RÉSULTAT DE L'IA :")
        print(result)
    else:
        print(f"Erreur : L'image {image_a_tester} n'a pas été trouvée.")