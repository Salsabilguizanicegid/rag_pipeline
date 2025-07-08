# 🧠 MODEL_INFO

## 📌 Purpose

This file documents the models used in the **RAG AI Assistant** project, their roles, how they work, and why they were chosen.

---

## 🔹 1. Sentence Transformer

- **Model used**: `all-MiniLM-L6-v2`  
- **Library**: [`sentence-transformers`]  
- **Purpose**: Converts each chunk of text extracted from the PDFs into numeric vectors for semantic search.  
- **Used in code**:

```python
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
vectors = embedding_model.encode(text_chunks)
```

## 🔹 2. FAISS (Facebook AI Similarity Search)

- **Purpose**: Quickly stores and searches text vectors to retrieve the most relevant chunks to a user query.
- **Index type used**: IndexFlatL2

```python
import faiss
index = faiss.IndexFlatL2(dimension)
index.add(vectors)
```
- **Why FAISS ?**
    Extremely fast for similarity search
    Maintained by Facebook AI Research
    Ideal for local RAG applications

## 🔹 3. Local LLM (eg, LLaMA 3 via Ollama)

- **Model used:** : llama3
- **Inference server** : Local API running at http://localhost:11434/api/generate (via Ollama)
- **Used in code:** :

```python
response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3", "prompt": prompt, "stream": False}
)
```
- **Prompt used** :
    You are an expert assistant. Answer the question using only the context provided.

    Context:
    ...

    Question: ...
    Answer:

## 🧩 Model Interaction Workflow
1. SentenceTransformer → Encodes text chunks into embeddings
2. FAISS → Retrieves the most semantically similar chunks to a query
3. LLM (LLaMA 3) → Generates an answer using only the retrieved chunks


[PDFs] → [Text] → [Chunks] → [Embeddings] → [FAISS Index]
                                                  ↓
                                              [User Question]
                                                  ↓
                                        [Top 3 Similar Chunks]
                                                  ↓
                                       [Prompt sent to the LLM]
                                                  ↓
                                             [Final Answer]
