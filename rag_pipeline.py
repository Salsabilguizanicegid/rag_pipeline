import os
import fitz  
import faiss
import numpy as np
import requests
import tiktoken
from typing import List, Optional
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def read_pdf(file_path: str) -> Optional[str]:
    if not os.path.isfile(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def chunk_text(text: str, max_tokens: int = 500) -> List[str]:
    if 'tiktoken' not in globals():
        print("tiktoken not available, splitting text by paragraphs.")
        return [para for para in text.split('\n\n') if para.strip()]
    
    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoder.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        decoded = encoder.decode(chunk)
        chunks.append(decoded)
    return chunks

def embed_texts(chunks: List[str]) -> np.ndarray:
    try:
        embeddings = embedding_model.encode(chunks, show_progress_bar=True, batch_size=32)
        return np.array(embeddings).astype("float32")
    except Exception as e:
        print(f"Error during embedding: {e}")
        return np.array([])

def create_faiss_index(vectors: np.ndarray) -> Optional[faiss.IndexFlatL2]:
    if vectors.size == 0:
        print("No vectors to index.")
        return None
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    print(f"FAISS index created with {index.ntotal} vectors.")
    return index

def save_faiss_index(index: faiss.IndexFlatL2, path: str) -> None:
    try:
        faiss.write_index(index, path)
        print(f"FAISS index saved to {path}")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

def load_faiss_index(path: str) -> Optional[faiss.IndexFlatL2]:
    if not os.path.isfile(path):
        print(f"No FAISS index found at {path}")
        return None
    try:
        index = faiss.read_index(path)
        print(f"FAISS index loaded from {path}")
        return index
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

def query_llama_local(context: str, question: str, model_name: str = "llama3") -> Optional[str]:
    prompt = f"""Tu es un assistant expert. Réponds à la question uniquement à partir du contexte fourni.

Contexte :
{context}

Question : {question}
Réponse :
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=10  
        )
        response.raise_for_status()
        data = response.json()
        if "response" in data:
            return data["response"].strip()
        else:
            print("Warning: 'response' key not in API response.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

def search_and_respond(query: str, chunks: List[str], index: faiss.IndexFlatL2, vectors: np.ndarray) -> Optional[str]:
    if index is None or vectors.size == 0:
        print("Invalid FAISS index or empty vectors.")
        return None
    query_vector = embedding_model.encode([query]).astype("float32")
    D, I = index.search(query_vector, k=3)
    retrieved_chunks = []
    for i in I[0]:
        if i < len(chunks):
            retrieved_chunks.append(chunks[i])
    context = "\n\n".join(retrieved_chunks)
    return query_llama_local(context, query)

if __name__ == "__main__":
    pdf_path = r"C:\Users\salsabil.guizani\OneDrive - CEGID\Desktop\Salsabil AI-POD\guide_fiscal_entreprise.pdf"
    index_path = "faiss_index.idx"

    text = read_pdf(pdf_path)
    if text is None:
        exit(1)

    chunks = chunk_text(text)
    print(f"Number of chunks created: {len(chunks)}")

    index = load_faiss_index(index_path)
    if index is None:
        vectors = embed_texts(chunks)
        if vectors.size == 0:
            exit(1)
        index = create_faiss_index(vectors)
        if index is None:
            exit(1)
        save_faiss_index(index, index_path)
    else:
        vectors = None

    while True:
        try:
            question = input("Pose ta question (ou tape 'exit' pour quitter) >> ").strip()
            if question.lower() == 'exit':
                print("Fin du programme.")
                break
            if len(question) == 0:
                print("Question vide, réessaie.")
                continue

            if vectors is None:
                print("Attention : embeddings non chargés, recherche impossible.")
                continue

            answer = search_and_respond(question, chunks, index, vectors)
            if answer:
                print("\nRéponse générée :\n")
                print(answer)
            else:
                print("Aucune réponse générée ou erreur lors de la requête.")
        except KeyboardInterrupt:
            print("\nProgramme interrompu par l'utilisateur.")
            break
