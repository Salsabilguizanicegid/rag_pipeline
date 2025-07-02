import os
import fitz  
import faiss  
import numpy as np
import requests
import tiktoken
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def read_pdf(file_path):
    if not os.path.isfile(file_path):
        print("Fichier non trouvé.")
        return None
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, max_tokens=500):
    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoder.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def embed_texts(chunks):
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings).astype("float32")

def get_faiss_index(vectors, index_path):
    if os.path.isfile(index_path):
        index = faiss.read_index(index_path)
        print(f"Index FAISS chargé depuis {index_path}")
    else:
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        faiss.write_index(index, index_path)
        print(f"Index FAISS créé et sauvegardé dans {index_path}")
    return index

def query_llama_local(context, question, model_name="llama3"):
    prompt = f"""Tu es un assistant expert. Réponds à la question uniquement à partir du contexte fourni.

Contexte :
{context}

Question : {question}
Réponse :
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=60  
        )
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.ReadTimeout:
        return "Le modèle a mis trop de temps à répondre. Veuillez réessayer."
    except requests.exceptions.RequestException as e:
        return f"Erreur de requête : {e}"


def search_and_respond(query, chunks, index):
    query_vector = embedding_model.encode([query]).astype("float32")
    D, I = index.search(query_vector, k=3)
    retrieved_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
    context = "\n\n".join(retrieved_chunks)
    return query_llama_local(context, query)

if __name__ == "__main__":
    pdf_path = r"C:\Users\salsabil.guizani\OneDrive - CEGID\Desktop\Salsabil AI-POD\guide_fiscal_entreprise.pdf"
    index_path = "faiss_index.idx"

    text = read_pdf(pdf_path)
    if text is None:
        exit()

    chunks = chunk_text(text)
    print(f"Nombre de chunks : {len(chunks)}")

    vectors = embed_texts(chunks)
    index = get_faiss_index(vectors, index_path)

    while True:
        question = input("Pose ta question (ou 'exit' pour quitter) >> ").strip()
        if question.lower() == "exit":
            print("Fin du programme.")
            break
        if not question:
            print("Question vide, essaie encore.")
            continue
        answer = search_and_respond(question, chunks, index)
        if answer:
            print("\nRéponse générée :\n")
            print(answer)
        else:
            print("Pas de réponse trouvée.")
