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
- Extremely fast for similarity search
- Maintained by Facebook AI Research
- Ideal for local RAG applications

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

## 🔹 4. Translation Module

- **Library used:** : [deep_translator]
- **Translator** :  GoogleTranslator
- **Language detection:** : [langdetect]
- **Purpose:** :
    - Automatically translates the user query to English before semantic search (if necessary).
    - Translates the final answer back to the original query language.

```python
from deep_translator import GoogleTranslator
from langdetect import detect

query_lang = detect(query)
translated_query = GoogleTranslator(source='auto', target='en').translate(query)
...
translated_response = GoogleTranslator(source='en', target=query_lang).translate(response)
```

## 🧩 Model Interaction Workflow
0. Translation Layer → Detects the user's query language and translates it to English for better semantic matching
1. SentenceTransformer → Encodes text chunks into embeddings
2. FAISS → Retrieves the most semantically similar chunks to a query
3. LLM (LLaMA 3) → Generates an answer using only the retrieved chunks
4. Back-Translation → Translates the generated answer back into the user’s original language

[PDF Files]
   ↓
[Text Extraction]
   ↓
[Text Chunking (200 words)]
   ↓
[SentenceTransformer → Embeddings]
   ↓
[FAISS Index]
   ↓
[User Query]
   ↓
[Language Detection + Translation to English]
   ↓
[Top 3 Relevant Chunks from FAISS]
   ↓
[Prompt sent to LLaMA 3 (via Ollama)]
   ↓
[Generated Answer]
   ↓
[Translate Answer to User’s Language]
   ↓
[Final Response]

