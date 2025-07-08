from rag_pipeline import RAGPipeline

if __name__ == "__main__":
    pdf_folder = r"C:\Users\salsabil.guizani\OneDrive - CEGID\Desktop\Salsabil AI-POD\pdfs"
    index_path = "faiss_index.idx"

    pipeline = RAGPipeline(pdf_folder, index_path)
    if not pipeline.prepare_pipeline():
        print("Erreur : impossible de charger les documents PDF.")
        exit()

    while True:
        question = input("Pose ta question (ou 'exit' pour quitter) >> ").strip()
        if question.lower() == "exit":
            print("Fin du programme.")
            break
        if not question:
            print("Question vide, essaie encore.")
            continue

        answer = pipeline.search_and_respond(question)
        print("\nRéponse générée :\n")
        print(answer or "Aucune réponse trouvée.")
