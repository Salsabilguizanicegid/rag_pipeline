# 🧠 MODEL_INFO.md

## 📌 Objectif

Ce fichier documente les modèles utilisés dans le projet **RAG AI Assistant**, leur rôle, leur fonctionnement et pourquoi ils ont été choisis.

---

## 🔹 1. Sentence Transformer

- **Modèle utilisé** : `all-MiniLM-L6-v2`
- **Bibliothèque** : [`sentence-transformers`](https://www.sbert.net/)
- **Fonction** : Convertir chaque chunk de texte issu des PDF en vecteurs numériques pour permettre la recherche sémantique.
- **Utilisation dans le code** :

```python
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
vectors = embedding_model.encode(text_chunks)
```

## 🔹 2. FAISS (Facebook AI Similarity Search)

Rôle : Stocker et rechercher rapidement les vecteurs de texte pour retrouver les documents les plus proches d'une question.
Type d'index utilisé : IndexFlatL2

```python
import faiss
index = faiss.IndexFlatL2(dimension)
index.add(vectors)
```
- **Pourquoi FAISS ?**
    Ultra rapide en recherche de similarité
    Maintenu par Facebook AI Research
    Parfait pour du RAG local

## 🔹 3. LLM Local (ex: LLaMA 3 via Ollama)

- **Modèle appelé** : llama3
- **Serveur d'inférence** : API locale disponible à http://localhost:11434/api/generate (via Ollama)
- **Méthode d'appel dans le code** :

```python
response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3", "prompt": prompt, "stream": False}
)
```
- **Prompt utilisé** :
    Tu es un assistant expert. Réponds à la question uniquement à partir du contexte fourni.

    Contexte :
    ...

    Question : ...
    Réponse :

- **Pourquoi ce choix ?**
    Permet de travailler sans envoyer les données sur internet
    Précis, fluide, personnalisable
    Facilement remplaçable par d'autres modèles compatibles

## 🧩 Interaction entre les modèles
1. SentenceTransformer → encode les morceaux de texte
2. FAISS → retrouve les chunks les plus proches d’une question
3. LLM (LLaMA 3) → génère la réponse en se basant uniquement sur les chunks récupérés


[PDFs] → [Texte] → [Chunks] → [Embeddings] → [FAISS_Index]
                                                   ↓
                                               [Question]
                                                   ↓
                                        [Top3_Chunks_similaires]
                                                   ↓
                                        [Prompt_envoyé_au_LLM]
                                                   ↓
                                              [Réponse_finale]
