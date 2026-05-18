from dotenv import load_dotenv
from crewai import Crew
from src.agents import DocumentAgents
from src.tasks import DocumentTasks
from src.tools import DocClassificationTool

load_dotenv()

# 1. Initialiser les composants
agents_factory = DocumentAgents()
tasks_factory = DocumentTasks()
classifier_tool = DocClassificationTool()

# 2. Créer les agents
classifier_agent = agents_factory.visual_classifier()
# On ajoute l'outil PyTorch à l'agent !
classifier_agent.tools = [classifier_tool]

extractor_agent = agents_factory.extractor_agent()

# 3. Définir le workflow
image_a_tester = "data/raw/test/invoice/sample_1.tif" # Un exemple
task1 = tasks_factory.classification_task(classifier_agent, image_a_tester)
task2 = tasks_factory.extraction_task(extractor_agent, "resultat.json")

# 4. Lancer la Crew
crew = Crew(
    agents=[classifier_agent, extractor_agent],
    tasks=[task1, task2],
    verbose=True
)

if __name__ == "__main__":
    print("--- Lancement du Système Multi-Agents ---")
    result = crew.kickoff()
    print("\n\n### RÉSULTAT FINAL ###")
    print(result)