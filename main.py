import os
import re
from crewai import Crew, Process
from src.agents import DocumentAgents
from src.tasks import DocumentTasks
from src.tools import DocClassificationTool, DocOCRTool
from dotenv import load_dotenv
load_dotenv()

def run_analysis(image_path: str):
    agents_factory = DocumentAgents()
    tasks_factory = DocumentTasks()
    classifier_tool = DocClassificationTool()
    ocr_tool = DocOCRTool()

    # --- OCR ---
    ocr_text = ocr_tool._run(image_path)
    print(f"\n📄 OCR extrait : {ocr_text[:200]}...\n")

    # --- Classification ---
    classification_result = classifier_tool._run(image_path)
    print(f"\n🔍 Classification : {classification_result}")

    # --- Parser la confiance ---
    confidence = float(classification_result.split("Confiance : ")[1].split(" |")[0])
    category = classification_result.split("Catégorie : ")[1].split(" |")[0]

    # --- HUMAN-IN-THE-LOOP si confiance < 80% ---
    if confidence < 0.80:
        print("\n" + "="*60)
        print("  ⚠  VALIDATION HUMAINE REQUISE")
        print(f"  Catégorie détectée : {category}")
        print(f"  Confiance : {confidence:.2%} (< 80%)")
        print("="*60)
        print("\nOptions :")
        print("  [1] Confirmer la catégorie détectée")
        print("  [2] Corriger manuellement")

        choix = input("\nVotre choix (1/2) : ").strip()

        if choix == "2":
            print(f"\nClasses disponibles : budget, email, form, handwritten, invoice, letter, memo, news_article, resume")
            category = input("Entrez la bonne catégorie : ").strip()
            print(f"✅ Catégorie corrigée : {category}")
        else:
            print(f"✅ Catégorie confirmée : {category}")

        from src.utils import log_action
        log_action("Humain", "HITL_validation", f"catégorie={category}, confiance={confidence}", "human_validated")

    # --- Suite : Extraction + Supervision ---
    extractor_agent = agents_factory.extractor_agent()
    extractor_agent.tools = [ocr_tool]
    supervisor_agent = agents_factory.supervisor_agent()

    task2 = tasks_factory.extraction_task(extractor_agent, ocr_text, category)
    task3 = tasks_factory.supervision_task(supervisor_agent)

    crew = Crew(
        agents=[extractor_agent, supervisor_agent],
        tasks=[task2, task3],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()  # ← result d'abord

    # --- Afficher le résumé ---
    print("\n" + "="*60)
    print("  📋 RÉSUMÉ DU DOCUMENT")
    print("="*60)
    try:
        result_str = str(result)
        if "resume_global" in result_str:
            match = re.search(r'"resume_global"\s*:\s*"([^"]+)"', result_str)
            if match:
                print(f"\n{match.group(1)}")
            else:
                print("\nRésumé disponible dans le JSON ci-dessus.")
        else:
            print("\nAucun résumé généré.")
    except:
        print("\nRésumé non extractible.")

    return result  # ← return à la fin


if __name__ == "__main__":
    image_a_tester = "data/raw/news_article/0000020424.tif"
    if os.path.exists(image_a_tester):
        print(f"\n{'='*60}")
        print(f"  Analyse : {image_a_tester}")
        print(f"{'='*60}\n")
        result = run_analysis(image_a_tester)
        print("\n" + "="*60)
        print("  RÉSULTAT FINAL")
        print("="*60)
        print(result)
    else:
        print(f"Erreur : fichier introuvable → {image_a_tester}")
