#Contenu : La définition de tes agents (Rôles, Backstories, Goals).
#Rôle : Tu y crées l'Agent Classificateur, l'Agent Extracteur, et l'Agent Superviseur.

from crewai import Agent, LLM
import os

class DocumentAgents:
    def __init__(self):
        # Utilisation de gemini-1.0-pro (plus compatible avec l'API v1beta)
        self.gemini_llm = LLM(
            model="gemini/gemini-1.0-pro",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2
        )

    def visual_classifier_agent(self):
        return Agent(
            role='Classificateur Visuel',
            goal='Identifier la catégorie d\'un document à partir de son image',
            backstory='Expert en analyse d\'images de documents, tu utilises des modèles de Deep Learning pour classer les documents.',
            llm=self.gemini_llm,
            verbose=True,
            allow_delegation=False
        )

    def extractor_agent(self):
        return Agent(
            role='Extracteur d\'Information NLP',
            goal='Lire le texte des documents et extraire les données clés en format JSON',
            backstory='Expert en compréhension de langage naturel, tu sais extraire des montants, des dates et des entités avec précision.',
            llm=self.gemini_llm,
            verbose=True,
            allow_delegation=False
        )