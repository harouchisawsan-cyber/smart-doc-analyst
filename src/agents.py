import os
from crewai import Agent, LLM


def get_ollama_llm():
    """Retourne le LLM local via Ollama"""
    return LLM(
        model="ollama/llama3.2",
        base_url="http://localhost:11434",  # URL locale d'Ollama
        temperature=0.2
    )


class DocumentAgents:
    def __init__(self):
        self.llm = get_ollama_llm()

    def visual_classifier_agent(self):
        """Agent 1 — Spécialiste classification visuelle (utilise le CNN PyTorch)"""
        return Agent(
            role='Classificateur Visuel',
            goal=(
                "Identifier avec précision la catégorie d'un document "
                "en utilisant le modèle de Deep Learning fourni."
            ),
            backstory=(
                "Tu es un expert en vision par ordinateur. "
                "Tu utilises un CNN ResNet18 entraîné sur des milliers de documents "
                "pour reconnaître instantanément les factures, emails, lettres, etc. "
                "Tu fournis toujours la catégorie détectée et le chemin de l'image analysée."
                "Si le score de confiance est faible (< 80%) ou si les données extraites "
                "semblent incomplètes, tu déclenches un point de contrôle humain. "
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def extractor_agent(self):
        """Agent 2 — Spécialiste extraction d'informations NLP"""
        return Agent(
            role="Extracteur d'Information NLP",
            goal=(
                "Extraire les données clés du document classifié "
                "et les retourner sous forme de JSON structuré."
            ),
            backstory=(
                "Tu es un expert en traitement du langage naturel. "
                "En fonction du type de document (facture, email, lettre...), "
                "tu sais exactement quelles informations extraire : "
                "montants, dates, noms, numéros de référence, etc. "
                "Tu produis toujours un JSON propre et bien structuré."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def supervisor_agent(self):
        """Agent 3 — Orchestrateur / Quality Supervisor (Human-in-the-Loop)"""
        return Agent(
            role='Superviseur Qualité et redacteur de synthese',
            goal=(
                "Consolider les résultats des deux agents spécialistes, "
                "vérifier leur cohérence, et décider si une validation humaine est nécessaire."
                 "et generer le rapport markdown finale"
            ),
            backstory=(
                "Tu es le chef d'orchestre du système. "
                "Tu reçois la classification et l'extraction, tu vérifies que tout est cohérent. "
                "Si le score de confiance est faible (< 80%) ou si les données extraites "
                "semblent incomplètes, tu déclenches un point de contrôle humain. "
                "Sinon, tu produis le rapport final consolidé "
                "tu rend les resultats elegants et pro "

            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )
