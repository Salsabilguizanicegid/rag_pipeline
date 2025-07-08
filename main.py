from rag_pipeline import RAGPipeline

if __name__ == "__main__":
    pdf_folder = r"C:\Users\salsabil.guizani\OneDrive - CEGID\Desktop\Salsabil AI-POD\pdfs"
    index_path = "faiss_index.idx"

    pipeline = RAGPipeline(pdf_folder, index_path)
    if not pipeline.prepare_pipeline():
        print("Error: unable to load PDF documents.")
        exit()

    while True:
        question = input("Ask your question (or type 'exit' to quit) >> ").strip()
        if question.lower() == "exit":
            print("Exiting the program.")
            break
        if not question:
            print("Empty question, please try again.")
            continue

        answer = pipeline.search_and_respond(question)
        print("\nGenerated Answer:\n")
        print(answer or "No answer found.")
