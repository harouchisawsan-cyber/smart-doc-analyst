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
                "confiance et categorie "
            ),
            agent=agent
        )
    
    def validation_task(self, agent, context):
        return Task(
            description=(
                "ARRÊT OBLIGATOIRE. Analyse les résultats de l'expert vision ci-dessus. "
                "Tu dois présenter la catégorie et la confiance à l'humain. "
                "NE DONNE PAS de réponse finale tant que l'humain n'a pas validé ou corrigé."
            ),
            expected_output="La catégorie confirmée et validée par l'opérateur humain et la confiance.",
            agent=agent,
            context=[context],
            human_in_the_loop=True
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

    def supervision_task(self, agent, confidence: float = 0.0) -> Task:
        return Task(
           description=(
            f"Score de confiance de la classification : {confidence}\n\n"
            "Tu reçois les résultats des deux agents précédents.\n"
            "Effectue les vérifications suivantes :\n"
            "1. La catégorie détectée est-elle cohérente avec les données extraites ?\n"
            "2. Les champs obligatoires sont-ils tous présents ?\n"
            "3. Le score de confiance est-il >= 0.80 ?\n\n"
            "RÈGLE HUMAN-IN-THE-LOOP :\n"
            "- Si confiance < 0.80 : statut = 'VALIDATION_HUMAINE_REQUISE'\n"
            "- Sinon : statut = 'APPROUVÉ_AUTOMATIQUEMENT'\n\n"
            "Utilise file_writer_tool pour sauvegarder le rapport dans 'rapportfinal.md'.\n"
            "Le contenu du fichier doit être un rapport markdown avec :\n"
            "- Le statut\n"
            "- La catégorie et la confiance\n"
            "- Le resume_global\n"
            "- Les données extraites\n"
        ),
        expected_output="Confirmation que rapportfinal.md a été créé avec le statut et le résumé.",
        agent=agent
    )
