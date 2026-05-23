from crewai import Task


class DocumentTasks:

    def classification_task(self, agent, image_path: str) -> Task:
        """Tâche 1 — Classification visuelle par le CNN"""
        return Task(
            description=(
                f"Analyse le document situé à : {image_path}\n"
                "Utilise l'outil Document_Classifier_Tool pour identifier la catégorie du document.\n"
                "Retourne : la catégorie détectée et le chemin du fichier analysé."
            ),
            expected_output=(
                "Une phrase indiquant la catégorie du document. "
                "Exemple : 'Le document data/raw/test/invoice/xxx.tif "
                "a été classifié comme : invoice'"
            ),
            agent=agent
        )

    def extraction_task(self, agent, ocr_text: str, category: str = "inconnu") -> Task:
        return Task(
            description=(
                f"Le document a été classifié comme : {category}\n\n"
                "Voici le texte extrait du document par OCR :\n\n"
                f"{ocr_text}\n\n"
                "En fonction du type détecté, extrais les informations clés :\n"
                "- Si INVOICE/BUDGET : Montant total, Date, Fournisseur, Numéro\n"
                "- Si EMAIL/MEMO/LETTER/HANDWRITTEN : Expéditeur, Destinataire, Date, Sujet, Résumé\n"
                "- Si FORM : champs remplis et leurs valeurs\n"
                "- Si NEWS_ARTICLE : Titre, Auteur, Date, Résumé\n"
                "- Si RESUME : Nom, Compétences, Expérience\n\n"
                "IMPORTANT : Pour TOUS les types de documents, ajoute toujours un champ "
                "'resume_global' qui résume le document en 2-3 phrases simples en français.\n\n"
                "Retourne un JSON structuré avec les vraies données."
            ),
            expected_output="Un JSON valide avec les champs adaptés au type de document.",
            agent=agent
        )

    def supervision_task(self, agent) -> Task:
        """Tâche 3 — Supervision, vérification cohérence + Human-in-the-Loop"""
        return Task(
            description=(
                "Tu reçois les résultats des deux agents précédents : "
                "la classification et l'extraction JSON.\n\n"
                "Effectue les vérifications suivantes :\n"
                "1. La catégorie détectée est-elle cohérente avec les données extraites ?\n"
                "2. Les champs obligatoires sont-ils tous présents ?\n"
                "3. Le score de confiance est-il >= 0.80 ?\n\n"
                "RÈGLE HUMAN-IN-THE-LOOP :\n"
                "- Si confiance < 0.80 OU si des champs critiques sont manquants : "
                "indique 'VALIDATION_HUMAINE_REQUISE' et liste les problèmes détectés.\n"
                "- Sinon : indique 'APPROUVÉ_AUTOMATIQUEMENT' et produis le rapport final.\n\n"
                "RÈGLE IMPORTANTE : si la confiance est inférieure à 0.80, "
                "tu DOIS obligatoirement mettre statut = 'VALIDATION_HUMAINE_REQUISE'.\n\n"
                "Produis un rapport JSON final consolidé."
            ),
            expected_output=(
                "Un rapport JSON final avec les champs : "
                "statut (APPROUVÉ_AUTOMATIQUEMENT ou VALIDATION_HUMAINE_REQUISE), "
                "classification, extraction, problemes_detectes (liste vide si aucun), "
                "rapport_final."
            ),
            agent=agent,
           
        )
