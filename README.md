# 🔍 RAG AI Assistant

Ce projet est un assistant de questions-réponses basé sur des documents PDF. Il utilise une approche de **Retrieval-Augmented Generation (RAG)** pour lire les documents, les indexer, et générer des réponses grâce à un LLM local.

## 📂 Structure

- `rag_pipeline.py` : cœur de la logique RAG (PDF ➜ texte ➜ chunks ➜ embeddings ➜ recherche ➜ réponse)
- `main.py` : interface en ligne de commande (CLI) pour interroger les documents.
- `app.py` : interface web avec Streamlit.
- `pdfs/` : dossier pour déposer les fichiers PDF.
- `faiss_index.idx` : fichier auto-généré contenant l'index FAISS.

## ⚙️ Installation

1. Cloner le projet :

```bash
git clone https://github.com/Salsabilguizanicegid/rag_pipeline/blob/main/rag_pipeline.py
cd rag_pipeline
```

2. Installer les dépendances : 

```bash
pip install -r requirements.txt
```

3. Placer vos fichiers PDF dans le dossier pdfs/.

## ▶️ Utilisation

1. En mode ligne de commande (test rapide):

```bash
python main.py
```

2. En mode interface web (Streamlit):
```bash
streamlit run app.py
```





