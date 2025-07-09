import os
import fitz  
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
import langdetect 

class RAGPipeline:
    def __init__(self, pdf_folder: str, index_path: str):
        self.pdf_folder = pdf_folder
        self.index_path = index_path
        self.text = ""
        self.chunks = []
        self.vectors = None
        self.index = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def read_pdfs(self):
        if not os.path.isdir(self.pdf_folder):
            print("Folder not found.")
            return

        full_text = ""
        for filename in os.listdir(self.pdf_folder):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(self.pdf_folder, filename)
                print(f"Reading file: {filename}")
                try:
                    doc = fitz.open(file_path)
                    for page in doc:
                        full_text += page.get_text()
                    doc.close()
                except Exception as e:
                    print(f"Error while reading {filename}: {e}")
        self.text = full_text

    def chunk_text(self, max_words=200):
        words = self.text.split()
        self.chunks = [
            " ".join(words[i:i + max_words])
            for i in range(0, len(words), max_words)
        ]

    def embed_chunks(self):
        self.vectors = self.embedding_model.encode(self.chunks, show_progress_bar=True)
        self.vectors = np.array(self.vectors).astype("float32")

    def build_or_load_index(self):
        if os.path.isfile(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"FAISS index loaded from {self.index_path}")
        else:
            dim = self.vectors.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.vectors)
            faiss.write_index(self.index, self.index_path)
            print(f"FAISS index created and saved to {self.index_path}")

    def search_and_respond(self, query: str, model_name="llama3"):
        try:
            query_lang = langdetect.detect(query)
        except:
            query_lang = "en"  # default fallback
        # Translate the query to English for semantic search (optional, or choose 'fr' if your docs are in French)
        translated_query = GoogleTranslator(source='auto', target='en').translate(query)
        # Embed and search with translated query
        query_vector = self.embedding_model.encode([translated_query]).astype("float32")
        D, I = self.index.search(query_vector, k=3)
        # Extract relevant chunks
        context = "\n\n".join(self.chunks[i] for i in I[0] if i < len(self.chunks))
        # Ask the model
        raw_response = self.query_llama_local(context, translated_query, model_name)
        # Translate the response back to the original language
        final_response = GoogleTranslator(source='en', target=query_lang).translate(raw_response)
        return final_response
    
    def query_llama_local(self, context: str, question: str, model_name="llama3"):
        prompt = f"""You are an expert assistant. Answer the question using only the context provided.

Context:
{context}

Question: {question}
Answer:
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
            return "The model took too long to respond. Please try again."
        except requests.exceptions.RequestException as e:
            return f"Request error: {e}"

    def prepare_pipeline(self):
        self.read_pdfs()
        if not self.text.strip():
            return False
        self.chunk_text()
        self.embed_chunks()
        self.build_or_load_index()
        return True
