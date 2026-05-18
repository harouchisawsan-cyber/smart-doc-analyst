#Contenu : La définition de tes agents (Rôles, Backstories, Goals).
#Rôle : Tu y crées l'Agent Classificateur, l'Agent Extracteur, et l'Agent Superviseur.

from crewai import Agent

# Agent de classification (Le spécialiste Deep Learning)
classifier = Agent(
  role='Document Classifier',
  goal='Classer les documents entrants en catégories précises',
  backstory='Expert en vision par ordinateur, tu analyses les images pour identifier le type de document.',
  verbose=True,
  allow_delegation=False
)