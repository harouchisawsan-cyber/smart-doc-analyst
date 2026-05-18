#Contenu : La définition de tes agents (Rôles, Backstories, Goals).
#Rôle : Tu y crées l'Agent Classificateur, l'Agent Extracteur, et l'Agent Superviseur.

from crewai import Agent

class DocumentAgents:
    def visual_classifier(self):
        return Agent(
            role='Classificateur Visuel',
            goal='Identifier le type de document via le modèle PyTorch',
            backstory='Expert en vision par ordinateur spécialisé en analyse documentaire.',
            verbose=True
        )