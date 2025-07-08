# 🔍 RAG AI Assistant

This project is a question-answering assistant based on PDF documents. It uses a **Retrieval-Augmented Generation (RAG)** approach to read documents, index them, and generate responses using a local LLM.

## 📂 Project Structure

- `rag_pipeline.py`: Core RAG logic (PDF ➜ text ➜ chunks ➜ embeddings ➜ search ➜ answer)
- `main.py`: Command-line interface (CLI) to query the documents.
- `app.py`: Web interface built with Streamlit.
- `pdfs/`: Folder to drop your PDF files.
- `faiss_index.idx`: Auto-generated file containing the FAISS index.

## ⚙️ Installation

1. Clone the project:

```bash
git clone https://github.com/Salsabilguizanicegid/rag_pipeline
cd rag_pipeline
```

2. Install the dependencies : 

```bash
pip install -r requirements.txt
```

3. Place your PDF files in the pdfs/.

## ▶️ Usage

1. Command-line mode (quick test):

```bash
python main.py
```

2. Web interface mode (Streamlit):
```bash
streamlit run app.py
```





